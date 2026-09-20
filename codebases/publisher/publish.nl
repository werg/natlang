---
description: Compose a cited document from pinned passages and prepare local Markdown and HTML.
args:
  brief: Text
  span_ids: Text[]
  collection_revision: Text
  table_ids: Text[]
  asset_ids: Text[]
  target: Text
returns: PublishReport
---
function publish(brief, span_ids, collection_revision, table_ids, asset_ids, target) -> PublishReport
  passages = read(span_ids, collection_revision)
  outline = plan(brief, passages, table_ids, asset_ids)
  document = compose(brief, outline, passages, collection_revision, table_ids, asset_ids)
  checked = check(document)
  if checked.ok:
    return prepare(document, target)
  return reject(target, checked)
