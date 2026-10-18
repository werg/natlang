# Checkpoint metadata lifetime

The supervisor chooses when to continue or recover training under the standing
adaptive production policy. This unit provides exact memory ownership plumbing;
it changes no training objective, model arithmetic, source admission or history.

1. Restore the full checkpoint, optimizer/RNG, source identities and schedule
   through the existing trainer. These are existing functions, unchanged.
2. Finish all setup decisions that read the predecessor, including pending
   admission, capacity checks and completed foundation cohorts.
3. Release setup-only references to the decoded predecessor and temporary
   restored frames before the first update. This is one crisp assignment block:
   Python local variables otherwise retain several large independent maps for
   the entire run. Keep the live source/index, counters and owned carried
   progress intact. Never mutate the predecessor's dictionaries to release them.
4. Save through the existing owned snapshot and atomic checkpoint writer.
   Snapshot independence, numeric tensors, optimizer/RNG, exposure and the
   checkpoint format remain unchanged.
5. Verify with a real checkpoint resume that setup containers die before model
   execution, while continuation and checkpoint round-trip retain exact credits.
   An isolated measurement of the real LFM14952 checkpoint measures memory;
   it grants no model, capacity or checkpoint-completion qualification.

Evidence: the old LFM process died after opening an empty checkpoint.pending.
The exact private14952 full state produced a15.50GiB host metadata-only high-water
mark with the standard owned snapshot and real PyTorch pickle path, excluding
model/source-index allocations. Its data.pkl buffer was1,135,003,485 bytes.
The current sandbox has less than5GiB global headroom, so its old save path is
not assumed safe. A controlled recovery may lose explicitly counted unsaved
updates; the existing verified full checkpoint and actual admitted source state
remain authoritative. Do not inject code into the running learner or claim an
unperformed final save.
