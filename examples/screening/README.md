# Screening

Screen job applications for a role. Every judgment is a function value that the program builds, passes and applies:

- **meets.nl**: `(requirement, application) => boolean` with `readout: decision`. `meets.partial(requirement)` is the
  check for one requirement: the same function with its first parameter bound, of type `(application) => boolean`.
- **fit_for.nl**: `(role) => (application: Application) => boolean`. Its result is a function: the call writes the fit
  judgment for that role (an inline `nl` whose instructions say what the role asks for), and the program applies it to
  every application.
- **screen.ts**: builds the list of checks, runs every check on every application at once, and does the bookkeeping
  (who is shortlisted, who misses which requirement) in code.
- **types.ts**: an application's `text` is `Untrusted<string>`, so the model reads it as quoted data; `Check` is the
  function type every check has.

```sh
natlang run examples/screening
```

A function value that crosses a typed slot is checked by its signature: passing `meets` itself where a `Check` is
expected is refused (`function-signature-mismatch`), because it takes two parameters and the slot passes one.
