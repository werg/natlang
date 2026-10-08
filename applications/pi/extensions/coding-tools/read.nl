---
description: pi's read tool. A bounded, line-addressed view of one text file (pi-durable tools/read.ts).
args:
  args: "{ path: string, offset?: number, limit?: number }"
returns: ToolExecutionResult
---
Read the text file args.path for the model, from line args.offset (1-based, default the first line) and at most
args.limit lines when a limit is given. The env service reaches the file; selection, in your folder, computes the
exact result. When a step below says "fail with", finish with status failed and that text exactly as the reason; it
becomes the tool's error.

1. path = env.readPath(args.path). On error, fail with error.message.
2. reader = env.openReader(path). On error, fail with error.message.
3. Read it, allowing one retry, because a concurrent writer can change the file between the line scan and the reads.
   For attempt 1 and, when needed, attempt 2:
   a. before = env.readerInfo(reader).
   b. result = selection(reader, before, args.path, args.offset, args.limit). Pass args.path, the path as the model
      wrote it: the messages quote it. If selection throws, fail with its message.
   c. after = env.readerInfo(reader).
   d. The read stands when after.size > before.size (the file only grew), or when after.size equals before.size and
      after.mtimeMs equals before.mtimeMs (unchanged). Then go to step 4 with result.
   e. Otherwise, after attempt 2, fail with "<args.path> changed while it was read".
   An error from readerInfo fails with its error.message.
4. env.closeReader(reader), whichever way the steps ended (also before failing once the reader is open).
5. Return result unchanged: it already holds the content, the diagnostics and the details.
