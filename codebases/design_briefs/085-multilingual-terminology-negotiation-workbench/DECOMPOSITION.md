# 085 — Multilingual Terminology Negotiation Workbench

## Distinct algorithm: negotiate a concept boundary first, then stage a coherent glossary migration

The workbench models a term occurrence as a language form in a specific product context and grammar, not as a dictionary string. It groups occurrences by concept only where their definitions and usage contexts align, retaining regional variants and multiple senses. Experts' disagreement about a concept boundary remains an open issue; popularity or frequency cannot settle meaning. Term options bind language, grammatical forms, register, region and usage condition.

A decision is versioned with the concept, evidence and authorized terminology owner. When a decision affects released content, propagation constructs an impact graph and stages a migration plan; it does not mix old and new terms in one workflow. Context-aware inflections are generated as candidates and reviewed before inclusion. The resume checkpoint tracks open disagreements, term forms, accepted decisions and affected content versions so a migration can pause and continue coherently.
