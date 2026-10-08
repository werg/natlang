---
description: pi's edit tool. Exact text replacements in one file, then pi's diff for UIs (pi-durable tools/edit.ts).
args:
  args: "{ path: string, edits: { oldText: string, newText: string }[] }"
returns: ToolExecutionResult
---
Apply args.edits to the file args.path. Your folder has text (byte-order mark and line endings) and applyEdits (the
matching policy). When a step says "fail with", finish with status failed and that text exactly as the reason. An
access error is "Could not edit file: <args.path>. Error code: <error.code>.", with args.path as the model wrote it.

1. Validate: args.edits must be a list with at least one edit; otherwise fail with
   "Edit tool input is invalid. edits must contain at least one replacement."
2. path = env.toolPath(args.path). On error, fail with error.message.
3. held = env.lock(path): edits and writes of one file in this process take turns. On error, fail with
   error.message. Release it with env.unlock(held.value) before returning or failing from here on.
4. info = env.fileInfo(path). On error, fail with the access error. info.value.kind other than "file" or "symlink"
   fails with "Could not edit file: <args.path>. Path is not a file."
5. original = env.readTextFile(path). On error, fail with the access error.
6. { bom, text: withoutBom } = text.stripBom(original.value); ending = text.detectLineEnding(withoutBom);
   lf = text.normalizeToLF(withoutBom).
7. applied = applyEdits(lf, args.edits, args.path). If it returns error, fail with that error exactly.
8. final = bom + text.restoreLineEndings(applied.content, ending). env.writeFile(path, final); on error, fail with the
   access error.
9. env.unlock(held.value). details = env.renderDiff(args.path, applied.base, applied.content).
10. Return { content: [{ type: "text", text: "Successfully replaced <args.edits.length> block(s) in <args.path>." }],
    details }.
