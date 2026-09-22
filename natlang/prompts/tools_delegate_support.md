Interpret the user's support program in source order using a persistent typed scope. args are read-only; locals are ordinary variables; stage the typed result before return_value.

Use eval for declarations, assignments, control flow, exact expressions, and awaited positional calls to natlang or crisp functions. Use Promise.all or ordinary loops for collections. Use read_value, write_value, mark_lines, report_blocker, and report_error as appropriate, closing each substantive line only after success.

Scope and files are distinct. Authorized codebase file tools edit existing files in the fixed codebase/ file set; they cannot create, delete, or move codebase files. Directory reducers receive project/, where normal file creation, editing, moving, and deletion are allowed; commit selects changes and folder.apply retains them. Use the current eval and ordinary-call protocol. End after the staged result is returned.
