# Shared checkpoint mapping readahead

The supervisor chooses to improve restore I/O after the actual Mellum48GB
continuation spent roughly twelve minutes restoring through virtiofs. The
observed live mapping had no remaining open descriptor, so advising a new file
descriptor would not change its later mapped faults. A bounded diagnostic read
measured175MB/s; it ran after SGD resumed and does not prove a restore speedup.

1. Keep the existing shared `CheckpointPolicy.load` and mmap/weights-only path.
2. After loading metadata, identify readable non-executable file-backed memory regions by
   exact filesystem device and inode of the checkpoint, rather than matching
   paths, guessing addresses or walking tensors and forcing faults.
3. Give only those regions Linux `MADV_SEQUENTIAL`. This is a kernel hint,
   with no eager whole-file prefetch, anonymous copy, thread or durable state.
   Unsupported platforms and advisory failures preserve normal loading.
4. An explicit shared environment setting can disable the hint for a workload
   whose observed access pattern does not benefit. Invalid settings fail before
   loading. No model, parameter, optimizer, RNG or checkpoint format changes.
5. Verify real mapped-region advice and tensor/optimizer/RNG equality against
   the disabled path; qualify actual sandbox behavior before claiming improved
   restore throughput. This does not replace existing host-state detachment.

The policy is the operational choice above. `/proc` parsing and `madvise` are
crisp OS plumbing. Reuse the shared checkpoint loader; no model-specific fork.

Root local qualification:18 tests passed, including a real Linux mapping whose
`VmFlags` acquired `sr`, exact hardlink device/inode selection, unrelated-file
exclusion, private-map isolation, advisory failure, disable/invalid-setting
handling, and bitwise next AdamW update/optimizer/RNG equality. No actual
full-size sandbox restore speedup is claimed.

Both actual training sandboxes also passed the tiny exact helper probe on
Torch2.14.0a0+b2c75dd062.nv26.09:20,480 mapped bytes genuinely gained the
sequential kernel flag, values/RNG/private-file isolation stayed exact, and
CUDA allocation remained zero. Exact source SHA and both receipts are in
`/home/werg/data/dgxops/checkpoint-readahead-cloud-qualification-root-20261016.json`.
This is target OS/loader mechanism evidence, not full-restore timing evidence.
