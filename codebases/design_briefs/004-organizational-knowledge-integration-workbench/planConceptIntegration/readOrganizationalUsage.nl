---
args: {"inquiry": "IntegrationInquiry"}
returns: "OperationalSense[]"
---
Extract term uses from actual source records, instructions and completion examples. For each organization/context bind senses = distinguishLocalSenses(source, inquiry.projects). Preserve former organizational names and provenance as aliases rather than rewriting historical records. Scope acronym expansions by actual context; two systems sharing a spelling remain distinct. Record actor roles, generated/consumed records and what completion means. Missing examples remain unresolved rather than replaced by glossary assertions.
