# Native weights-only views for off-machine task evaluation

The production recipe decides where routine evaluation runs. This mechanism
must let evaluation consume the exact trained model without ferrying its Adam
moments or holding up a training GPU. Preserve full resumable parents unchanged.

1. Extend the existing verified checkpoint-copy mechanism to recognize the
   native checkpoint's nested training optimizer. Drop only optimizer entries.
2. Keep every backbone/interface tensor, dtype, configuration, step and source
   fact exact. Verify the written values bit for bit and publish atomically.
3. Record the full source checkpoint SHA separately from the derived view SHA.
   A derived view is never the same artifact as the full checkpoint.
4. Refuse this view in the default training loader. The serving engine explicitly
   opts into weights-only loading and reports the original-parent provenance.

Existing private export, independently verified transfer, async task scheduler,
physical runtime snapshots and ledger admission remain shared mechanisms. A
local view test is not provider-transfer or full-model serving qualification.
