# Outline

Digest a nested document, section by section. The recursion follows the data, and every result is checked against
what it claims:

- **digest.nl** declares `recursive: structural`: it sees itself as `digest` and calls itself on each subsection (a
  part of its argument), all at once. A self-call on anything that is not smaller is refused when it happens.
- Its `ensures:` holds every digest, at every level, to two postconditions before its caller sees it: the exact one in
  code (`covers(result, section)`: the same title, one part per subsection, in order), and a judged one (the gist states
  the main point of the section's own text), which the judge reads with `section` beside the digest. A result that
  fails goes back to the call with the clause's message, and the call repairs it.
- **digest/covers.ts** is the exact check. Because a `check:` names it, it judges the call: `edit_code` refuses it.
- **types.ts**: a section's `text` is `Untrusted<string>`, so the model and the judge read it as quoted data.

```sh
natlang run examples/outline
```
