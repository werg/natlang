---
args: {"segment": "ContextSegment", "locale": "string", "glossary": "EvidenceItem[]"}
returns: "ConceptLocaleTerm[]"
---
For each concept and sense used by this segment, resolve the locale-specific approved form and grammatical role from versioned authority. Preserve regional variants and scoped exceptions. Conflicting approved terms require a named owner question; frequency in existing translations cannot override authority. Keep a product concept's terminology independent from unrelated senses of the same source word.
