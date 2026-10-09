---
description: Migration site location. Widen one search hit to the whole statement or declaration around it, with the exact lines.
args:
  intent: Intent
  hit: SearchHit
  revision: string
returns: CheckedSite
---
Locate the site of hit at revision. A site is the smallest run of whole lines that holds a complete statement,
declaration or import around the hit, and it spans several lines when the statement does (a call with its arguments
on following lines, a function signature, an import list).

1. context = repository.lines(hit.path, max(1, hit.line - 5), hit.line + 5, revision). Read it to see where the
   statement that contains the hit begins and ends.
2. from and to are the first and last line (1-based, in the file) of that statement. At most 20 lines; a longer
   construct is cut at the line that holds the hit and the lines it needs.
3. text = repository.lines(hit.path, from, to, revision). Take the text from the service as it is.

Return { id: "", path: hit.path, from, to, text, hits: 1 }.
