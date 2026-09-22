---
args:
  request: string
  snapshot: RepoSnapshot
  uses: SearchResult
returns: Patch[]
---
Plan exact old/new replacements for the requested migration, covering every
relevant caller. Each old snippet must appear exactly once in its file.
Search hits are evidence to inspect, not automatic edits. Preserve behavior
and acceptance tests. Return only patches in the declared file manifest.
If checks fail, inspect their output and propose a new candidate revision.
