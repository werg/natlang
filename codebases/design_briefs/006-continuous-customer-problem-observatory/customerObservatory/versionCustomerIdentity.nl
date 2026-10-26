---
args: { episodes: EpisodeRegister, mappings: EvidencePage[] }
returns: IdentityVersionLedger
---
Link migrated customer identifiers only through a versioned, source-backed identity mapping with effective interval. Preserve prior IDs and mapping versions. If a person cannot be reliably matched, keep the incident identity unresolved and block customer-level counts across the migration.
