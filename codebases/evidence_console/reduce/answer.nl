---
description: Search a versioned collection, read exact passages, and compose a cited answer.
args:
  question: string
returns: EvidenceAnswer
---
function answer(question) -> EvidenceAnswer
  queries = plan_search(question)
  found = search(queries)
  selected = select(question, found)
  passages = read(selected, found.collection_revision)
  draft = compose(question, passages, found.truncated)
  return verify(passages, draft, found.collection_revision)
