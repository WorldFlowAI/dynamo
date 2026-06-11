// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic KV donor lifecycle events.
//!
//! The contract by which Dynamo-side emitters (engine adapters, KVBM, the
//! mocker) tell external semantic KV providers what reusable KV exists, where
//! it lives, and when it stops existing, so providers can maintain a
//! fleet-scale donor catalog with correct lifecycle without scraping engines.
//!
//! Two planes (see the internal design doc):
//!
//! - **Lifecycle spine (this contract, generic, content-free).** Carries
//!   identity, structured compatibility, location, segment shape, and
//!   lifecycle. No tokens, no embeddings.
//! - **Semantic enrichment (provider-owned).** A segment's embedding (or PQ
//!   codes, sparse signature, ...) rides the opaque [`DonorSegment::provider_metadata`]
//!   field, attached by the engine-local adapter that observed the tokens and
//!   keyed by the same [`DonorRegistered::donor_id`]. Dynamo never parses it.
//!   Kept opaque on purpose: representation is a provider choice, so a typed
//!   field would couple Dynamo to one provider.
//!
//! Load-bearing design rules:
//!
//! - **Identity is global and stable; location is separate and mutable.**
//!   [`DonorRegistered::donor_id`] is a globally-unique handle (the join key for
//!   the whole contract). Where the KV physically lives is a [`DonorLocation`]
//!   that can change (G1 -> G2 offload, move to a shared store) without changing
//!   the donor's identity. This is what lets the contract describe fleet-scale,
//!   shared-tier, and external-store donors, not just worker-pinned ones.
//! - **Multi-segment from the start.** A donor exposes one or more
//!   [`DonorSegment`]s; a whole-sequence donor is the one-segment degenerate
//!   case. Eviction can target the whole donor or specific segments.
//! - **Separate plane, versioned envelope, recoverable streams.** Events travel
//!   on [`SEMANTIC_KV_EVENT_SUBJECT`], never as new variants of the exact
//!   plane's closed `KvCacheEventData`. `event_id` is monotonic per emitter
//!   stream `(worker_id, dp_rank)`; consumers detect gaps and converge to
//!   fail-closed, and bootstrap via a snapshot (mirrors the exact indexer's
//!   dump/recovery).
//! - **Liveness-bounded.** Donors located on a worker are purged when that
//!   worker is removed (same liveness source as the exact indexer's
//!   `remove_worker`), covering crash-without-evict. See [`DonorsPurged`].

use std::collections::{BTreeMap, VecDeque};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::protocols::{
    DpRank, ExternalSequenceBlockHash, LocalBlockHash, StorageTier, WorkerId, compute_block_hash,
};

/// Transport subject for semantic KV events, parallel to `kv-events`.
pub const SEMANTIC_KV_EVENT_SUBJECT: &str = "semantic-kv-events";

/// Current envelope schema version. Bump on any incompatible change.
pub const SEMANTIC_KV_EVENT_SCHEMA_VERSION: u16 = 1;

/// Default per-stream retention in [`SemanticEventBuffer`].
pub const DEFAULT_SEMANTIC_EVENT_BUFFER_SIZE: usize = 1024;

/// Versioned envelope for one semantic KV event. `worker_id`/`dp_rank` identify
/// the **emitter stream** (for ordering and gap detection), which is distinct
/// from where a donor's KV lives ([`DonorLocation`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticKvEvent {
    pub schema_version: u16,
    /// Monotonic per `(worker_id, dp_rank)` emitter stream, starting at 0.
    pub event_id: u64,
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub data: SemanticKvEventData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SemanticKvEventData {
    /// One or more segments of a donor became reusable. Availability at
    /// emission time, not a reservation. Re-emitting an existing `donor_id`
    /// updates it (e.g. new location after offload, added segments).
    DonorRegistered(DonorRegistered),
    /// Some or all of a donor's segments are no longer reusable.
    DonorEvicted(DonorEvicted),
    /// Bulk invalidation by scope, used for liveness (a worker died without
    /// emitting per-donor evictions) and operational resets.
    DonorsPurged(DonorsPurged),
    /// Cache flush/reset on this emitter: every donor it announced with an
    /// older generation is stale.
    ProviderGenerationReset { generation: u64 },
}

/// Where reusable KV physically resides. Structured for the cases Dynamo can
/// reason about (reachability, transfer-cost class); the exact store address is
/// opaque for external/shared backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum DonorLocation {
    /// Resident on a specific worker, at a specific memory tier.
    Worker {
        worker_id: WorkerId,
        dp_rank: DpRank,
        tier: StorageTier,
    },
    /// In a store reachable independently of any single worker (e.g.
    /// Mooncake/HiCache, KVBM external/object tier). `backend` names the store
    /// kind; `locator` is the store-specific address, opaque to Dynamo and
    /// resolved by the connector that owns that backend.
    Shared {
        backend: String,
        tier: StorageTier,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        locator: Vec<u8>,
    },
}

impl DonorLocation {
    /// The owning worker, if this location is pinned to one. `None` for shared
    /// stores. Used by liveness purge to decide what a worker's removal drops.
    pub fn owning_worker(&self) -> Option<WorkerId> {
        match self {
            Self::Worker { worker_id, .. } => Some(*worker_id),
            Self::Shared { .. } => None,
        }
    }

    pub fn tier(&self) -> StorageTier {
        match self {
            Self::Worker { tier, .. } | Self::Shared { tier, .. } => *tier,
        }
    }
}

/// Structured compatibility key. The **hard subset** (`model`, `tokenizer`,
/// `kv_layout`, `block_size`) must match for KV bytes to be reusable at all;
/// `lora`/`quant`/`extra` let a provider reason about partial compatibility
/// (e.g. base-layer reuse across LoRAs) instead of all-or-nothing. Dynamo
/// guarantees only the hard subset; finer policy is provider/engine territory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheNamespace {
    pub model: String,
    pub tokenizer: String,
    /// Attention backend + dtype + head/layout identity that fixes KV byte
    /// compatibility. KV from one `kv_layout` is not reusable under another.
    pub kv_layout: String,
    pub block_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Forward-compatibility for fields not yet modeled. Part of the hard key:
    /// unknown keys present on one side and absent on the other are a mismatch.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, String>,
}

impl CacheNamespace {
    /// Whether KV under `self` can be reused for a request whose namespace is
    /// `other`, ignoring adapter/quant nuance (the hard, byte-compatibility
    /// subset). A provider applies its own `lora`/`quant` policy on top.
    pub fn hard_compatible_with(&self, other: &Self) -> bool {
        self.model == other.model
            && self.tokenizer == other.tokenizer
            && self.kv_layout == other.kv_layout
            && self.block_size == other.block_size
            && self.extra == other.extra
    }

    /// Strict reuse compatibility: hard subset plus exact adapter and quant.
    pub fn fully_compatible_with(&self, other: &Self) -> bool {
        self.hard_compatible_with(other) && self.lora == other.lora && self.quant == other.quant
    }
}

/// A reusable span of a donor sequence and its per-segment enrichment. Semantic
/// matching is segment-level, so the embedding lives here, not on the donor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DonorSegment {
    /// Stable id within the donor, referenced by [`DonorEvicted::segments`].
    pub segment_id: u32,
    /// `[start, end)` token positions in the donor sequence.
    pub token_range: (u32, u32),
    /// Non-reversible integrity/dedup digest over the segment's token ids. Not
    /// a semantic key: it cannot be used for similarity search.
    pub digest: LocalBlockHash,
    /// Optional join key to the exact event plane's block lineage for this span.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_hashes: Option<Vec<ExternalSequenceBlockHash>>,
    /// Optional provider-specific enrichment (embedding / PQ codes / sparse
    /// signature). Opaque to Dynamo; only the provider reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DonorRegistered {
    /// Globally-unique, stable donor identity. The join key for the whole
    /// contract: it MUST equal the id an engine-local adapter uses when it
    /// enriches the donor, and the id later [`DonorEvicted`] events carry.
    /// Identity does not change when the donor's [`DonorLocation`] changes.
    pub donor_id: Uuid,
    pub namespace: CacheNamespace,
    /// Current placement. Re-registering the same `donor_id` with a new
    /// location expresses an offload/move.
    pub location: DonorLocation,
    pub token_count: u32,
    /// One or more reusable segments. A whole-sequence donor is a single
    /// segment spanning `[0, token_count)`. Must be non-empty.
    pub segments: Vec<DonorSegment>,
    pub provider_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DonorEvicted {
    pub donor_id: Uuid,
    /// `None` evicts the whole donor; `Some` evicts only the listed
    /// `segment_id`s (e.g. a partial block eviction that leaves other segments
    /// reusable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<u32>>,
    pub provider_generation: u64,
}

/// Bulk invalidation. The liveness path: when Dynamo removes a worker (lease
/// expiry / disconnect, the same signal that drives the exact indexer's
/// `remove_worker`), an emitter publishes `DonorsPurged { scope: Worker(..) }`
/// so a semantic-only consumer drops that worker's donors without waiting for
/// per-donor evictions that a crashed worker never sent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DonorsPurged {
    pub scope: PurgeScope,
    pub provider_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PurgeScope {
    /// Every donor whose location's owning worker is this worker.
    Worker { worker_id: WorkerId },
    /// Every donor at this exact location (e.g. a shared store going offline).
    Location { location: DonorLocation },
    /// Every donor from this emitter.
    All,
}

/// Non-reversible digest over a token sequence, using the same hashing family
/// as the exact event plane so an engine-local provider can reproduce it from
/// tokens it observed. Integrity/dedup only, never a semantic key.
pub fn sequence_digest(tokens: &[u64]) -> LocalBlockHash {
    let mut bytes = Vec::with_capacity(tokens.len() * 8);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    compute_block_hash(&bytes)
}

/// Outcome of applying one event to a [`SemanticEventBuffer`] stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOutcome {
    Applied,
    /// `event_id` below the stream cursor; already seen.
    Duplicate,
    /// Missing events: the stream's donors must be invalidated and replayed.
    Gap {
        expected: u64,
        got: u64,
    },
    /// Envelope from a future schema; dropped, never best-effort parsed.
    VersionUnsupported {
        got: u16,
    },
}

#[derive(Debug, Default)]
struct StreamState {
    next_event_id: u64,
    events: VecDeque<SemanticKvEvent>,
    /// Set on gap; cleared by [`SemanticEventBuffer::acknowledge_invalidation`]
    /// once the consumer has dropped the stream's donors and replayed.
    invalidated: bool,
}

/// Consumer-side ordered buffer with gap detection, bounded replay, and
/// snapshot bootstrap, mirroring the exact plane's local-indexer semantics.
#[derive(Debug)]
pub struct SemanticEventBuffer {
    max_events_per_stream: usize,
    streams: FxHashMap<(WorkerId, DpRank), StreamState>,
}

impl Default for SemanticEventBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_SEMANTIC_EVENT_BUFFER_SIZE)
    }
}

impl SemanticEventBuffer {
    pub fn new(max_events_per_stream: usize) -> Self {
        Self {
            max_events_per_stream: max_events_per_stream.max(1),
            streams: FxHashMap::default(),
        }
    }

    /// Apply one event in arrival order.
    pub fn apply(&mut self, event: SemanticKvEvent) -> ApplyOutcome {
        if event.schema_version != SEMANTIC_KV_EVENT_SCHEMA_VERSION {
            return ApplyOutcome::VersionUnsupported {
                got: event.schema_version,
            };
        }
        let stream = self
            .streams
            .entry((event.worker_id, event.dp_rank))
            .or_default();

        if event.event_id < stream.next_event_id {
            return ApplyOutcome::Duplicate;
        }
        if event.event_id > stream.next_event_id {
            let expected = stream.next_event_id;
            // Resynchronize past the gap; the consumer must drop this stream's
            // donors and replay (or re-snapshot) before trusting it again.
            stream.invalidated = true;
            stream.events.clear();
            stream.next_event_id = event.event_id + 1;
            stream.events.push_back(event.clone());
            return ApplyOutcome::Gap {
                expected,
                got: event.event_id,
            };
        }

        stream.next_event_id = event.event_id + 1;
        stream.events.push_back(event);
        while stream.events.len() > self.max_events_per_stream {
            stream.events.pop_front();
        }
        ApplyOutcome::Applied
    }

    /// Seed a freshly-started consumer from a producer snapshot: the current
    /// live `DonorRegistered` set plus, per emitter stream, the `event_id` to
    /// resume tailing from (the snapshot watermark + 1). Mirrors fetching the
    /// exact indexer's dump and then subscribing past it. Returns the snapshot
    /// donor events for the caller to load into its catalog.
    pub fn apply_snapshot(
        &mut self,
        watermarks: impl IntoIterator<Item = (WorkerId, DpRank, u64)>,
        donors: Vec<SemanticKvEvent>,
    ) -> Vec<SemanticKvEvent> {
        for (worker_id, dp_rank, next_event_id) in watermarks {
            let stream = self.streams.entry((worker_id, dp_rank)).or_default();
            stream.next_event_id = next_event_id;
            stream.invalidated = false;
            stream.events.clear();
        }
        donors
    }

    /// Whether the stream saw a gap and has not been acknowledged since.
    pub fn is_invalidated(&self, worker_id: WorkerId, dp_rank: DpRank) -> bool {
        self.streams
            .get(&(worker_id, dp_rank))
            .is_some_and(|s| s.invalidated)
    }

    /// Consumer signal that it has dropped the stream's donors and replayed.
    pub fn acknowledge_invalidation(&mut self, worker_id: WorkerId, dp_rank: DpRank) {
        if let Some(stream) = self.streams.get_mut(&(worker_id, dp_rank)) {
            stream.invalidated = false;
        }
    }

    /// Buffered events with ids in `[start_id, end_id]`, or `None` when the
    /// range is not fully buffered or the stream is invalidated; callers must
    /// then re-snapshot from the emitter.
    pub fn events_in_id_range(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_id: u64,
        end_id: u64,
    ) -> Option<Vec<SemanticKvEvent>> {
        if start_id > end_id {
            return Some(Vec::new());
        }
        let stream = self.streams.get(&(worker_id, dp_rank))?;
        if stream.invalidated {
            return None;
        }
        let oldest = stream.events.front()?.event_id;
        let newest = stream.events.back()?.event_id;
        if start_id < oldest || end_id > newest {
            return None;
        }
        Some(
            stream
                .events
                .iter()
                .filter(|e| e.event_id >= start_id && e.event_id <= end_id)
                .cloned()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace() -> CacheNamespace {
        CacheNamespace {
            model: "qwen2.5-7b".to_string(),
            tokenizer: "qwen2.5".to_string(),
            kv_layout: "flashinfer-fp16".to_string(),
            block_size: 16,
            lora: None,
            quant: None,
            extra: BTreeMap::new(),
        }
    }

    fn segment(id: u32, range: (u32, u32)) -> DonorSegment {
        DonorSegment {
            segment_id: id,
            token_range: range,
            digest: sequence_digest(&[id as u64, range.0 as u64, range.1 as u64]),
            block_hashes: None,
            provider_metadata: Some(vec![1, 2, 3]), // stand-in embedding payload
        }
    }

    fn registered_event(id: u64, donor: Uuid, segments: Vec<DonorSegment>) -> SemanticKvEvent {
        SemanticKvEvent {
            schema_version: SEMANTIC_KV_EVENT_SCHEMA_VERSION,
            event_id: id,
            worker_id: 7,
            dp_rank: 0,
            data: SemanticKvEventData::DonorRegistered(DonorRegistered {
                donor_id: donor,
                namespace: namespace(),
                location: DonorLocation::Worker {
                    worker_id: 7,
                    dp_rank: 0,
                    tier: StorageTier::Device,
                },
                token_count: 256,
                segments,
                provider_generation: 0,
            }),
        }
    }

    #[test]
    fn round_trips_multi_segment_through_json() {
        let original = registered_event(
            0,
            Uuid::nil(),
            vec![segment(0, (0, 128)), segment(1, (128, 256))],
        );
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains("\"kind\":\"donor_registered\""));
        let parsed: SemanticKvEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
        let SemanticKvEventData::DonorRegistered(reg) = parsed.data else {
            panic!("expected donor_registered");
        };
        assert_eq!(reg.segments.len(), 2);
    }

    #[test]
    fn shared_location_round_trips() {
        let loc = DonorLocation::Shared {
            backend: "mooncake".to_string(),
            tier: StorageTier::External,
            locator: vec![0xde, 0xad],
        };
        let json = serde_json::to_string(&loc).unwrap();
        let parsed: DonorLocation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, loc);
        assert_eq!(parsed.owning_worker(), None);
        assert_eq!(parsed.tier(), StorageTier::External);
    }

    #[test]
    fn namespace_hard_vs_full_compatibility() {
        let base = namespace();
        let mut lora_a = namespace();
        lora_a.lora = Some("adapter-a".to_string());
        // Different LoRA: byte-compatible base, not fully compatible.
        assert!(base.hard_compatible_with(&lora_a));
        assert!(!base.fully_compatible_with(&lora_a));
        // Different kv_layout: not even byte-compatible.
        let mut other_layout = namespace();
        other_layout.kv_layout = "trtllm-fp8".to_string();
        assert!(!base.hard_compatible_with(&other_layout));
    }

    #[test]
    fn future_schema_versions_are_rejected() {
        let mut buffer = SemanticEventBuffer::default();
        let mut ev = registered_event(0, Uuid::nil(), vec![segment(0, (0, 256))]);
        ev.schema_version = SEMANTIC_KV_EVENT_SCHEMA_VERSION + 1;
        assert_eq!(
            buffer.apply(ev),
            ApplyOutcome::VersionUnsupported {
                got: SEMANTIC_KV_EVENT_SCHEMA_VERSION + 1
            }
        );
    }

    #[test]
    fn in_order_applies_and_duplicates_are_flagged() {
        let mut buffer = SemanticEventBuffer::default();
        let seg = vec![segment(0, (0, 256))];
        assert_eq!(
            buffer.apply(registered_event(0, Uuid::nil(), seg.clone())),
            ApplyOutcome::Applied
        );
        assert_eq!(
            buffer.apply(registered_event(1, Uuid::nil(), seg.clone())),
            ApplyOutcome::Applied
        );
        assert_eq!(
            buffer.apply(registered_event(0, Uuid::nil(), seg)),
            ApplyOutcome::Duplicate
        );
        assert!(!buffer.is_invalidated(7, 0));
    }

    #[test]
    fn gaps_invalidate_until_acknowledged() {
        let mut buffer = SemanticEventBuffer::default();
        let seg = vec![segment(0, (0, 256))];
        assert_eq!(
            buffer.apply(registered_event(0, Uuid::nil(), seg.clone())),
            ApplyOutcome::Applied
        );
        assert_eq!(
            buffer.apply(registered_event(3, Uuid::nil(), seg.clone())),
            ApplyOutcome::Gap {
                expected: 1,
                got: 3
            }
        );
        assert!(buffer.is_invalidated(7, 0));
        assert_eq!(buffer.events_in_id_range(7, 0, 3, 3), None);

        buffer.acknowledge_invalidation(7, 0);
        assert_eq!(
            buffer.apply(registered_event(4, Uuid::nil(), seg)),
            ApplyOutcome::Applied
        );
        assert_eq!(
            buffer.events_in_id_range(7, 0, 3, 4).map(|v| v.len()),
            Some(2)
        );
    }

    #[test]
    fn snapshot_seeds_watermark_so_tailing_resumes_without_a_false_gap() {
        let mut buffer = SemanticEventBuffer::default();
        // Producer snapshot says stream (7,0) is current through event_id 41.
        let donors = buffer.apply_snapshot(
            [(7u64, 0u32, 42u64)],
            vec![registered_event(
                40,
                Uuid::nil(),
                vec![segment(0, (0, 256))],
            )],
        );
        assert_eq!(
            donors.len(),
            1,
            "snapshot donors handed back to the catalog"
        );
        // Tailing resumes at 42 with no spurious gap from 0.
        assert_eq!(
            buffer.apply(registered_event(
                42,
                Uuid::nil(),
                vec![segment(0, (0, 256))]
            )),
            ApplyOutcome::Applied
        );
        assert!(!buffer.is_invalidated(7, 0));
    }

    #[test]
    fn segment_and_whole_donor_eviction_shapes() {
        // Whole-donor eviction.
        let whole = DonorEvicted {
            donor_id: Uuid::nil(),
            segments: None,
            provider_generation: 0,
        };
        // Specific-segment eviction (partial block eviction).
        let partial = DonorEvicted {
            donor_id: Uuid::nil(),
            segments: Some(vec![1, 3]),
            provider_generation: 0,
        };
        for ev in [whole, partial] {
            let json = serde_json::to_string(&ev).unwrap();
            assert_eq!(serde_json::from_str::<DonorEvicted>(&json).unwrap(), ev);
        }
    }

    #[test]
    fn liveness_purge_scopes_round_trip() {
        let by_worker = SemanticKvEventData::DonorsPurged(DonorsPurged {
            scope: PurgeScope::Worker { worker_id: 7 },
            provider_generation: 0,
        });
        let by_location = SemanticKvEventData::DonorsPurged(DonorsPurged {
            scope: PurgeScope::Location {
                location: DonorLocation::Shared {
                    backend: "mooncake".to_string(),
                    tier: StorageTier::External,
                    locator: vec![],
                },
            },
            provider_generation: 0,
        });
        for data in [by_worker, by_location] {
            let json = serde_json::to_string(&data).unwrap();
            assert_eq!(
                serde_json::from_str::<SemanticKvEventData>(&json).unwrap(),
                data
            );
        }
    }

    #[test]
    fn streams_are_independent_per_worker_and_dp_rank() {
        let mut buffer = SemanticEventBuffer::default();
        let seg = vec![segment(0, (0, 256))];
        let mut other = registered_event(0, Uuid::nil(), seg.clone());
        other.worker_id = 8;
        assert_eq!(
            buffer.apply(registered_event(0, Uuid::nil(), seg.clone())),
            ApplyOutcome::Applied
        );
        assert_eq!(buffer.apply(other), ApplyOutcome::Applied);
        assert_eq!(
            buffer.apply(registered_event(5, Uuid::nil(), seg)),
            ApplyOutcome::Gap {
                expected: 1,
                got: 5
            }
        );
        assert!(buffer.is_invalidated(7, 0));
        assert!(!buffer.is_invalidated(8, 0));
    }

    #[test]
    fn sequence_digest_is_stable_and_content_sensitive() {
        let a = sequence_digest(&[1, 2, 3, 4]);
        assert_eq!(a, sequence_digest(&[1, 2, 3, 4]));
        assert_ne!(a, sequence_digest(&[1, 2, 3, 5]));
        assert_ne!(a, sequence_digest(&[1, 2, 3]));
    }
}
