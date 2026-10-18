# One full checkpoint verification at the controller boundary

The supervising policy is to resume useful training promptly after an earned update, while preserving exact continuation checks. Mellum's completed 20515 update waited with its GPU idle while the bridge client decoded its 48.3 GB checkpoint; the controller then decoded the same checkpoint again. Neither decode was a private backup transfer.

1. A standalone queue client continues to verify the complete result by default.
2. When the shared controller invokes the known queued Python bridge module, explicitly ask that child to check the request/result inventory only. The child still drains the exact worker and preserves cooperative stop handling.
3. Immediately after the child returns, perform the controller's existing complete artifact and semantic checkpoint verification. Do not serialize or trust a claimed verified checkpoint, skip this verification, or treat inventory validation as continuation readiness.
4. Keep custom provider commands unchanged. Reject the deferred child option outside queue mode. Tampered artifacts and invalid optimizer/RNG/parent histories must still fail at the controller boundary.
5. Execute the known queued client from the exact currently loaded bridge module through a SHA-bound subprocess bootstrap. Preserve the numerical PYTHONPATH when CPU control modules are overlaid independently. Verify the helper bytes before execution; an older installed CLI must not receive the new flag. No claimed verified checkpoint state crosses processes.

This is crisp command and verification plumbing for the existing production policy. No scheduling, admission, model-facing text, objective or checkpoint schema changes. The exact worker-result checks remain shared across models and raw/semantic rounds. Frozen cloud runtimes require an explicit code transition; publishing this source does not change an already running client.
