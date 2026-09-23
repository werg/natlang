---
args:
  request: string
  snapshot: RepoSnapshot
  uses: SearchResult
  failed?: Validation
returns: Patch[]
---
Plan exact old/new replacements for the requested migration, covering every
relevant caller. Each old snippet must appear exactly once in its file of the
snapshot revision. Search hits are evidence to inspect, not automatic edits.
Preserve behavior and acceptance tests. Return only patches in the declared
file manifest. When failed is present, the snapshot is the candidate that
failed those checks: read their output and propose the repair on top of it.
