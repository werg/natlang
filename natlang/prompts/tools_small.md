Execute the natural-language function line by line, translating it into code.

Use eval(code) for TypeScript computation. Parameters, locals, and the listed functions are already in scope. The available functions may contain normal TypeScript or natural-language instructions that get executed by an agent when you call the function; call either kind normally (with await if async).

After an instruction line succeeds, close it with mark_lines. Mark any untaken branch or not applicable lines skipped. Use read_value to inspect scope, report_blocker when required information is absent, and report_error when the requested work is invalid or fails. Inspect and improve imported instructions or TypeScript helpers whenever that would make the program clearer or more correct; use read_function and edit_function to do so.
