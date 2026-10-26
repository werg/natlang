---
args: { observation: FacilityObservation, aliases: FacilityLocationAlias[], locations: FacilityLocation[] }
returns: FacilityLocationResolution
---
Resolve an alias only when verified evidence maps it to one official location ID. Keep the alias attached for search and user communication; an ambiguous or unverified name blocks work-order dispatch until corrected by an accountable asset source.
