# Preserve a reviewed in-process owner's normal return

The supervisor chooses when to refresh training. Existing drain and recipe
handoff functions remain responsible for exact checkpoint, optimizer, RNG,
exposure, source, and runtime continuity.

The recovery wrapper writes its owner, launch binding, time and incomplete
recipe status only after the shared trainer returns normally. Its marker has a
different filename and shape from the original `entry-terminal.json` contract.
Extend that same crisp completion verifier with an explicitly selected second
contract. Capture it while the actual owner lives, before any terminal exists.
Bind and snapshot the original launch metadata, verify its owner and wrapper,
and require that exact metadata binding in the later marker. No invented OS
exit code, numerical qualification or recipe completion is produced.

Keep the original completion contract unchanged. Reject an unrelated marker,
late capture, changed wrapper/owner/launch, failure evidence, or completion
credit. Tests use real child processes and consume the normal return through
the existing verifier. The final drain still requires the owner to have exited
and an independently pinned checkpoint.

Owned reconciliation must also preserve earned work across the existing verified
metadata classification continuation. Reuse the recipe's exact continuation
validator; permit only that input role to differ. Normalize only its exclusive
metadata dependencies, retaining every data/held/other-role input identity.
Return the old earned stage contract and record future tuning without granting
future-stage completion. Genuine data additions must be adopted and trained
before this operational handoff; they cannot borrow old exposure here.
