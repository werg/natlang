---
args: {"use": "DatasetUse", "labels": "ObservableLabel[]", "families": "SourceFamily[]", "eligibility": "RecordEligibility[]", "accounting": "SliceAccounting[]", "session": "DatasetAuditSession", "prior": "DatasetEditionState"}
returns: "FrozenDatasetEdition | null"
---
Call verifyWorldSplitClosure(families, eligibility). Refuse release when a source/world family crosses train, dev or test, when identity remains disputed, or when required coverage and label observability remain blocked. Build the admitted record and artifact inventory only from actual supplied artifacts and current audience-specific rights. Bind the edition to use, annotation version, family assignments and exact owner approval. An unapproved prospective edition has released false; missing artifacts return null with concrete acquisition work, not a fabricated manifest.
