# Atomic state publication owns its temporary file

Before-edit inventory: common/jsonio.py implements write_bytes_durable and
write_json_durable. Twenty-three source files reference these APIs: the expert
iteration/remote collectors, held/replay/JOLT/memetic tools, learner and native
checkpoint/bundle publication, source builders and retention metadata. They all
share this Python filesystem primitive. Vibe campaign publish was a separate
fixed-.tmp implementation used by status, receipts, graph metrics and archive
health. Derivative recovery and the live collector can publish archive health
concurrently. Existing higher-level locks still own read/modify/write semantics
and immutable history; an atomic replace is not a merge transaction.

The defect is independent of JSON encoding: two writers can write the same
pending inode, then one rename removes the other's pathname, or one failure
cleanup deletes the other's pending output. Fix the shared primitive with an
exclusively created, operation-unique temporary file. Cleanup removes that file
only after this operation actually created it. Preserve file/directory fsync,
whole-file rename, encoding and ordinary file-creation permissions. UUID names
use OS entropy, not any training/sampling RNG. No output/schema randomness is
introduced. Vibe publish now uses that primitive with its original exact JSON
encoding, rather than retaining another implementation.

All Python engine manifest producers using this helper receive the fix. Managed
llama-server, in-page wllama and ts-host consume the resulting manifests; they do
not execute this Python publication API. Their own independent state writers are
not claimed audited or changed here. Large weight/optimizer binary publication
uses its existing ownership path; this change does not replace that protocol.

Controls force overlapping real writes, interleave a failed writer's cleanup
with a successful writer, and force a temporary-name collision. They require
whole unchanged output bytes and preservation of another owner's file. Existing
golden hashing/encoding tests remain applicable. A fresh campaign also now
creates its owned waves/jobs parents, rather than requiring an undocumented
precreated waves directory.
