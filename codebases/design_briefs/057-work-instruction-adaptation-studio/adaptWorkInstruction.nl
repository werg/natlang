---
args: { adaptation: InstructionAdaptationCase }
returns: InstructionReleasePacket
---
1. Bind ambiguity = resolveFloorTerminology(adaptation.localTerminology, adaptation.equipmentReferences). If a local term denotes multiple controls or an alias lacks verified asset identity, mark the affected instruction step blocked.
2. Bind warnings = preserveWarningMeaning(adaptation.translations). Any omitted hazard or altered condition blocks release of that warning's step.
3. Bind variant = assembleLocalVariant(adaptation.baseline, ambiguity, warnings, adaptation.localExceptions, adaptation.interruptionRules). An exception can waive only its named constraint; inspection controls remain explicit.
4. Bind replay = replayRestartAndExceptions(variant, adaptation.interruptionRules, adaptation.localExceptions). A mid-step interruption must resume from the defined safe state and repeat every required verification.
5. Bind withdrawal = enumerateFloorCopyWithdrawal(adaptation.floorCopies, adaptation.sourceVersion). Existing printed copies stay in scope until a recorded withdrawal task is assigned.
6. Return { variant, replay, withdrawalTasks: withdrawal, applicableCopyIds: adaptation.floorCopies.map(copy => copy.copyId), releasable: ambiguity.ambiguousTerms.length == 0 && warnings.safe && replay.failedSteps.length == 0 && replay.interruptionFailures.length == 0, approvalRequired: true }.
