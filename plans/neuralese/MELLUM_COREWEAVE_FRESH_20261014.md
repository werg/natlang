# Mellum production CoreWeave lineage, 2026-10-14

Start `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`
as a fresh full-weight lineage from the same clean Mellum base and raw-token
heads used by the local clean R8 warmup. This is an explicit supervision decision
under the owner's order to get both full production trainers onto CoreWeave ASAP.
The measured gateway upload of the 1.15 GB QAD checkpoint took 1,269 seconds;
using that route for Mellum's roughly47 GiB complete state would take about14
hours. Private artifact transport remains under investigation for subsequent
checkpoint exports and transfers. Do not describe this cloud start as a resume.

The sandbox `a7858948-0e1a-4c89-bb69-e2cc29cbc243` has a Blackwell RTX PRO6000
with96 GiB VRAM, four CPU cores and64 GiB requested host RAM (75.9 GB physical).
Root storage has about7.2 TB available. No scratch volume is mounted. NGC26.09
provides Torch2.14/CUDA13.4 and Muon. The official OCI manifest inspected during
preparation pins its amd64 image to
`sha256:f71fd67f6d71f685e0bd08ce6c8df48807624cc5d48886370d3471bc4f89d81d`.
Keep the actual sandbox image request and runtime package receipt alongside the
run; a requested image tag alone is not runtime digest attestation.

The recipe preserves the admitted full R8 input
`5a6d54f526c8259e255ae45408d9257c880bd5051c0bd092227cb65393af39c5`,
full backbone training, shared MuonSR, FP8 expert implementation, packed windows,
0.5-epoch warmup and synchronous held-task evaluation (reference13, noise1).
It pins the integrated shared Python inventory. Run the actual trainer's step-0
identity check, and capture FP8/Muon dispatch, memory calibration and measured
peak on the96 GiB device. Qualification still requires the existing held CE,
drift and task-success gates; a completed step count does not suffice.

Keep local DGX training productive until cloud updates and a full checkpoint
save have succeeded. Then stop the local unit gracefully, preserving its exact
checkpoint, optimizer and RNG as the migration fallback. Do not kill it merely
because a sandbox exists. The DGX owner performs that stop. PH and whole-trajectory
continuations need their independently admitted data and explicit lineage bindings.
The separately audited PH prefix is ready; the full R8 WT receipt remains in flight.
