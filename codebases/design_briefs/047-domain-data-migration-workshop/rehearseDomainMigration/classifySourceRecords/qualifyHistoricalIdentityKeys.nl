---
args: { records: SourceRecord[], mappings: DomainFieldMapping[] }
returns: HistoricalIdentityFinding[]
---
Group source identifiers by system and validity interval. If the same raw ID overlaps incompatible entity histories, preserve separate identity keys and their relationship evidence; block an unqualified merge until domain-owner review.
