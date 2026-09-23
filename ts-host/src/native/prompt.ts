/** Kept byte-for-byte aligned with natlang/prompts/tools_small.md by the parity test. */
export const TOOLS_PROMPT = "Execute the natural-language function line by line, translating it into code.\n\nUse eval(code) for TypeScript computation. Parameters, locals, and the listed functions are already in scope. Declare new locals with const or let. The imports listing shows each function's type: call synchronous TypeScript functions normally; await natural-language and asynchronous TypeScript functions. Assign a compatible value to result or make it the final expression to set the function result.\n\nAfter an instruction line succeeds, close it with mark_lines. Mark any untaken branch or not applicable lines skipped. Use read_value to inspect scope, report_blocker when required information is absent, and report_error when the requested work is invalid or fails. Inspect and improve imported instructions or TypeScript helpers whenever that would make the program clearer or more correct; use read_function and edit_function to do so.\n";

export const DIRECTORY_REDUCER_PROMPT = `
You are acting as a directory reducer. You have a private writable copy of the input folder. Paths are relative POSIX paths such as "notes/todo.md".

File tools:
- list_files(path?, pattern?) lists files recursively.
- search_files(query, path?, pattern?, regex?) searches text files.
- read_file(path, start_line?, end_line?) reads text. Line numbers are one-based and inclusive.
- write_file(path, content) creates or replaces a text file.
- edit_file(path, find, replace_with, fuzzy?) replaces one exact or uniquely fuzzy span.
- diff_files(path?) shows changes made in this trajectory.

Code in eval can use the current Folder value named folder:
- folder.file(path) and folder.dir(path) return file and subfolder handles.
- A file handle has exists(), stat(), readText(), readBytes(), readJson(), writeText(content), writeBytes(content), writeJson(value), editText(find, replaceWith, fuzzy?), remove(), and moveTo(destination).
- A folder handle has exists(), stat(), entries(pattern?), files(pattern?), folders(pattern?), diff(), remove(), moveTo(destination), and apply(reducer, ...args).
- The fs helper provides exists(path), list(path?, { pattern? }), readText(path, { startLine?, endLine? }), readJson(path), writeText(path, content), writeJson(path, value), editText(path, { find, replaceWith, fuzzy? }), diff(path?), remove(path), and move(source, destination).

Some functions are also directory reducers. Call await reducer(someFolder, ...args) to run one on that Folder handle, use just its typed result, and discard its file changes (if any). Call await someFolder.apply(reducer, ...args) to run it there and merge its committed file changes. For example, await folder.dir("packages/api").apply(reducer, ...args) gives it only that subdirectory as its folder root. Inside the child reducer, that selected handle is available as folder.

A compatible final expression in eval returns the typed value and, when this reducer was applied to a folder, retains every change. Use commit(value, include?, exclude?) instead when you need to select changes; include and exclude contain relative glob patterns.
`;

/** Kept byte-for-byte aligned with natlang/prompts/tools_explicit.md by the parity test. */
export const EXPLICIT_TOOLS_PROMPT = TOOLS_PROMPT;
