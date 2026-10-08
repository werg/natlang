---
description: pi's write tool. Create or overwrite one file, creating parent directories (pi-durable tools/write.ts).
args:
  args: "{ path: string, content: string }"
returns: ToolExecutionResult
---
Write args.content to the file args.path. When a step says "fail with", finish with status failed and that text
exactly as the reason.

1. path = env.toolPath(args.path). On error, fail with error.message.
2. held = env.lock(path): writes and edits of one file in this process take turns. On error, fail with error.message.
3. env.writeFile(path, args.content). On error, env.unlock(held.value) and fail with error.message.
4. env.unlock(held.value).
5. Return { content: [{ type: "text", text: "Successfully wrote to <args.path>" }] }, with args.path as the model
   wrote it.
