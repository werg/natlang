---
description: Compose a cited document from pinned passages and prepare local Markdown and HTML. Read files for local asset or editorial context when needed.
args:
  brief: Text
  span_ids: Text[]
  collection_revision: Text
  table_ids: Text[]
  asset_ids: Text[]
  target: Text
  files?: Dict<File>
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
