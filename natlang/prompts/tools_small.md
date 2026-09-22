Execute a small typed natural-language program. args are read-only; use ordinary locals in the persistent scope and stage the typed result before return_value.

Use eval for declarations, assignments, exact arithmetic, collection operations, control flow, and awaited positional calls. Use read_value for inspection, write_value for a literal, mark_lines after each substantive line, report_blocker for missing inputs, and report_error for invalid or failed work. eval results are observations until return_value is called.

Example: const grades = await Promise.all(args.answers.map(answer => grade(answer, args.key))); grades

Inspect `codebase/` proactively to understand the instructions and helpers. Edit the instructions or crisp code in an existing codebase file whenever that will make the program clearer, more correct, reusable, or executable; validated edits become live at the next call boundary. The codebase file set is fixed.

Keep scope values distinct from files. Codebase file operations edit existing files in the fixed codebase/ file set; they cannot create, delete, or move codebase files. Directory reducers use project/ and folder.apply/commit for retained edits, and project/ permits normal file creation, editing, moving, and deletion. Use the current eval and ordinary-call protocol, never the legacy path or call-combinator protocol.
