import check from "./publish/check";
import compose from "./publish/compose";
import plan from "./publish/plan";
import prepare from "./publish/prepare";
import read from "./publish/read";
import reject from "./publish/reject";
---
description: Compose a cited document from pinned passages and prepare local Markdown and HTML. Read files for local asset or editorial context when needed.
args:
  brief: string
  span_ids: string[]
  collection_revision: string
  table_ids: string[]
  asset_ids: string[]
  target: string
  files?: Record<string, File>
returns: PublishReport
---
function publish(brief, span_ids, collection_revision, table_ids, asset_ids, target, files) -> PublishReport
  passages = read(span_ids, collection_revision)
  outline = plan(brief, passages, table_ids, asset_ids, files)
  document = compose(brief, outline, passages, collection_revision, table_ids, asset_ids, files)
  checked = check(document)
  if checked.ok:
    return prepare(document, target)
  return reject(target, checked)
