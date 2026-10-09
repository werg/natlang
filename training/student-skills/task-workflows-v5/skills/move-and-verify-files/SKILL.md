---
name: move-and-verify-files
description: Use when files must be moved or archived. Move source handles and verify the destination tree.
---

Identify the source path and destination before a write. Move the SOURCE handle: await folder.file("inbox/a.md").moveTo(folder.dir("archive")); then inspect folder.file("archive/a.md") and folder.diff(). Handles retain their original paths after moving; reacquire a handle at the destination. Do not move a handle already taken from the destination into that same destination. If several files qualify, finish all required changes and return precisely the requested value. Preserve unrelated files. Verify both the final tree and the return value.
