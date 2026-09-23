import compose from "./answer/compose";
import plan_search from "./answer/plan_search";
import read from "./answer/read";
import search from "./answer/search";
import select from "./answer/select";
import verify from "./answer/verify";
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
