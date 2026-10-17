# Verify task selection metadata before checkpoint loading

The existing curriculum policy admits only ordinary tasks or tasks bearing its
existing `qualified` label. Keep that policy unchanged. Invoke the same
`task_descriptors` verifier alongside the existing bound TRAIN/task-source
checks before entering the heavyweight model handoff verifier, including on a
cached configuration path. This is exact input plumbing, not a second selector.

Malformed qualification/level values must fail before any checkpoint is loaded.
Focused regressions exercise this ordering and retain ordinary/qualified cases.
The executing model sees no new prompts, errors or task instructions.
