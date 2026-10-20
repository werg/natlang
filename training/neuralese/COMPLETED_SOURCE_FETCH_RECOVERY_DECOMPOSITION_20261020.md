# Preserve completed updates when the next source fetch fails

The owner policy is to recover useful completed optimizer work at a safe update
boundary, while preserving the original source failure and every row obligation.
A rejected next batch earns no exposure. An optimizer failure is a different,
possibly partial boundary and must never enter this recovery path.

1. The shared supervised loop captures the existing Python/Torch/CUDA training
   RNG immediately before requesting its next scheduled batch.
2. Crisp exception handling covers only that request. It restores the captured
   RNG and, when this invocation completed updates, invokes the existing full
   `_save` path with the current model, optimizer, row bindings, exact completed
   exposure, cursor, FP8 state and carried stage progress.
3. The failure is metadata, never a successful training/final receipt. Save
   failure adds a note to the original exception; that exception is rethrown.
   RNG is restored after the save attempt too, including a failed save.
4. Parsing, CE validation, admission, the training update and all normal end
   checks keep their existing mechanisms. Both backbones share this one path.
5. Tiny real trainer checks compare completed checkpoints against uninterrupted
   reference parameters, optimizer moments, RNG and exposure on both models,
   exercise save failure and exact resume, and reject partial-update recovery.
