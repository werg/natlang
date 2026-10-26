---
args: { records: Evidence[], links: EvidencePage[] }
returns: StudyLineage
---
Link preprints, journal articles, protocols and corrections only when identifiers, authorship, methods and version provenance support the same underlying study. Preserve distinct report versions and identify true duplicate records; do not count a journal article as an independent new study merely because its publication status changed.
