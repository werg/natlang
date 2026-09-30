/** Narrow title-based MuSiQue path prototype with scoped path-reference updates. */
import { createHash } from 'node:crypto';

const sha256 = value => createHash('sha256').update(value).digest('hex');
const VARIANT_SUFFIX = ':title-path-v2-r2';
const REVISION = 'musique-title-paths/2026-09-30-v2-r2';
const EVIDENCE_PATH_LISTS = ['retrieved', 'world', 'background'];

function titleSlug(title) {
  const slug = title.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase()
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64).replace(/-+$/g, '');
  return slug || 'article';
}

function targetFor(sourcePath, title) {
  const match = /^articles\/(\d+)\.md$/.exec(sourcePath);
  if (!match) throw new Error(`musique_title_path_source_invalid:${sourcePath}`);
  const index = Number(match[1]);
  if (!Number.isSafeInteger(index) || index < 0 || match[1] !== String(index))
    throw new Error(`musique_title_path_index_invalid:${sourcePath}`);
  const path = `articles/${String(index).padStart(3, '0')}-${titleSlug(title)}.md`;
  if (path.length > 100) throw new Error(`musique_title_path_too_long:${path}`);
  return path;
}

function articleTitle(content, path) {
  if (typeof content !== 'string') throw new Error(`musique_title_path_content_invalid:${path}`);
  const firstLine = content.split(/\r?\n/, 1)[0];
  const match = /^#[\t ]+(.+)$/.exec(firstLine);
  if (!match) throw new Error(`musique_title_path_title_missing:${path}`);
  return match[1].trim();
}

function replaceExplicitPathStrings(value, pathMap) {
  if (typeof value === 'string') {
    let result = value;
    for (const [from, to] of Object.entries(pathMap).sort(([a], [b]) => b.length - a.length || a.localeCompare(b)))
      result = result.replaceAll(from, to);
    return result;
  }
  if (Array.isArray(value)) return value.map(item => replaceExplicitPathStrings(item, pathMap));
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, replaceExplicitPathStrings(item, pathMap)]));
  return value;
}

function remapReferenceArgs(value, pathMap, oldPaths, key = '') {
  if (key === 'value' || key === 'answer' || key === 'expected') return value;
  if (typeof value === 'string') {
    if (key === 'path') return pathMap[value] ?? value;
    if (key === 'code') {
      if (oldPaths.some(path => value.includes(path)))
        throw new Error('musique_title_path_unsupported_numeric_path_in_reference_code');
      return value;
    }
    return value;
  }
  if (Array.isArray(value)) {
    if (key === 'paths') return value.map(path => typeof path === 'string' ? pathMap[path] ?? path : path);
    return value.map(item => remapReferenceArgs(item, pathMap, oldPaths));
  }
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value).map(([childKey, item]) =>
      [childKey, remapReferenceArgs(item, pathMap, oldPaths, childKey)]));
  return value;
}

function remapReferenceCalls(reference, pathMap) {
  if (!reference || typeof reference !== 'object') return reference;
  const root = reference.root;
  if (!Array.isArray(root)) return reference;
  const oldPaths = Object.keys(pathMap);
  return { ...reference, root: root.map(call => Array.isArray(call) && call.length === 2 &&
    typeof call[0] === 'string' && call[1] && typeof call[1] === 'object' ?
    [call[0], remapReferenceArgs(call[1], pathMap, oldPaths)] : call) };
}

function hasStaleReferencePath(reference, oldPaths) {
  const contains = value => typeof value === 'string' && oldPaths.some(path => value.includes(path));
  const scanArgs = (value, key = '') => {
    if (key === 'value' || key === 'answer' || key === 'expected') return false;
    if (key === 'path' || key === 'code') return contains(value);
    if (key === 'paths' && Array.isArray(value)) return value.some(contains);
    if (Array.isArray(value)) return value.some(item => scanArgs(item));
    if (value && typeof value === 'object') return Object.entries(value).some(([childKey, item]) => scanArgs(item, childKey));
    return false;
  };
  return (reference?.root ?? []).some(call => Array.isArray(call) && call.length === 2 &&
    typeof call[0] === 'string' && scanArgs(call[1]));
}

function remapFileKeys(files, pathMap, field) {
  if (files == null) return files;
  if (typeof files !== 'object' || Array.isArray(files)) throw new Error(`musique_title_path_file_map_invalid:${field}`);
  const next = {};
  for (const [path, body] of Object.entries(files)) {
    const target = pathMap[path];
    if (path.startsWith('articles/') && !target) throw new Error(`musique_title_path_map_missing:${field}:${path}`);
    const nextPath = target ?? path;
    if (Object.hasOwn(next, nextPath)) throw new Error(`musique_title_path_collision:${field}:${nextPath}`);
    next[nextPath] = body;
  }
  return next;
}

function remapEvidencePathList(items, pathMap) {
  if (!Array.isArray(items)) return items;
  return items.map(item => {
    if (typeof item === 'string') return pathMap[item] ?? item;
    if (item && typeof item === 'object' && typeof item.path === 'string')
      return { ...item, path: pathMap[item.path] ?? item.path };
    return item;
  });
}

function validateVariant(record, review) {
  if (review?.registry !== REVISION || record.id !== `${review.base_ir_id}${VARIANT_SUFFIX}` ||
      review.source_id !== (record.external_source?.source_id ?? record.source_ids?.[0]) ||
      review.source_id !== record.source_ids?.[0] ||
      review.source_snapshot_sha256 !== (record.external_source?.snapshot_sha256 ?? null) ||
      JSON.stringify(review) !== JSON.stringify(record.generation?.article_path_review))
    throw new Error(`musique_title_path_variant_metadata_mismatch:${record.source_ids?.[0]}`);

  const map = review.path_map;
  const articles = review.articles;
  const fileMapEntries = Object.entries(map ?? {});
  const articlePaths = Object.keys(record.semantics?.folder_files ?? {}).filter(path => path.startsWith('articles/'));
  const targetSet = new Set(fileMapEntries.map(([, target]) => target));
  if (!Array.isArray(articles) || articles.length !== fileMapEntries.length ||
      articlePaths.length !== targetSet.size || articlePaths.some(path => !targetSet.has(path)))
    throw new Error(`musique_title_path_article_set_mismatch:${record.source_ids?.[0]}`);

  const articleBySource = new Map(articles.map(article => [article.source_path, article]));
  for (const [sourcePath, targetPath] of fileMapEntries) {
    const article = articleBySource.get(sourcePath);
    if (!article || article.target_path !== targetPath) throw new Error(`musique_title_path_map_forged:${sourcePath}`);
    const content = record.semantics.folder_files[targetPath];
    const title = articleTitle(content, targetPath);
    if (targetFor(sourcePath, title) !== targetPath || title !== article.title ||
        sha256(title) !== article.title_sha256 || sha256(content) !== article.content_sha256)
      throw new Error(`musique_title_path_article_integrity_mismatch:${sourcePath}`);
  }

  const expectedMap = review.expected_file_path_map;
  const expectedEntries = Object.entries(expectedMap ?? {});
  const expectedArticlePaths = Object.keys(record.semantics?.expected_files ?? {}).filter(path => path.startsWith('articles/'));
  const expectedTargets = new Set(expectedEntries.map(([, target]) => target));
  if (expectedEntries.length !== (review.expected_file_records?.length ?? 0) ||
      expectedArticlePaths.length !== expectedTargets.size || expectedArticlePaths.some(path => !expectedTargets.has(path)))
    throw new Error(`musique_title_path_expected_map_mismatch:${record.source_ids?.[0]}`);
  for (const [sourcePath, targetPath] of expectedEntries) {
    const item = review.expected_file_records.find(entry => entry.source_path === sourcePath && entry.target_path === targetPath);
    if (map[sourcePath] !== targetPath || !item || !(targetPath in record.semantics.expected_files) ||
        sha256(record.semantics.expected_files[targetPath]) !== item.content_sha256)
      throw new Error(`musique_title_path_expected_file_integrity_mismatch:${sourcePath}`);
  }

  if (review.retrieved_paths?.length !== (record.curriculum?.evidence?.retrieved ?? []).length ||
      JSON.stringify(review.retrieved_paths) !== JSON.stringify(record.curriculum.evidence.retrieved) ||
      record.curriculum.evidence.retrieved.some(path => !(path in record.semantics.folder_files)))
    throw new Error(`musique_title_path_retrieved_paths_mismatch:${record.source_ids?.[0]}`);
  for (const key of EVIDENCE_PATH_LISTS) {
    const expected = review.evidence_path_lists?.[key];
    const actual = record.curriculum?.evidence?.[key];
    if (expected !== undefined && JSON.stringify(expected) !== JSON.stringify(actual))
      throw new Error(`musique_title_path_evidence_paths_mismatch:${key}`);
    for (const item of actual ?? []) {
      const path = typeof item === 'string' ? item : item?.path;
      if (path?.startsWith('articles/') && !(path in record.semantics.folder_files))
        throw new Error(`musique_title_path_evidence_path_unresolved:${key}:${path}`);
    }
  }

  const oldPaths = Object.keys(map);
  const scopedInstructions = JSON.stringify(record.semantics.files);
  if (oldPaths.some(path => scopedInstructions.includes(path)) || hasStaleReferencePath(record.curriculum.reference, oldPaths))
    throw new Error(`musique_title_path_stale_reference:${record.source_ids?.[0]}`);
}

/**
 * Rename numeric MuSiQue article paths from their supplied Markdown H1 titles.
 * Apply reviewed source/oracle transforms first; their evidence keeps its
 * canonical source paths and this review's map resolves those paths to files.
 */
export function applyMusiqueTitlePathsV2(record) {
  if (record.source !== 'musique') return record;
  if (record.id.endsWith(VARIANT_SUFFIX)) {
    validateVariant(record, record.generation?.article_path_review);
    return record;
  }

  const files = record.semantics?.folder_files;
  if (!files || typeof files !== 'object' || Array.isArray(files))
    throw new Error(`musique_title_path_files_missing:${record.source_ids?.[0]}`);
  const sourceArticlePaths = Object.keys(files).filter(path => path.startsWith('articles/'));
  const pathMap = {};
  const targets = new Set();
  const articles = [];
  for (const sourcePath of sourceArticlePaths) {
    const title = articleTitle(files[sourcePath], sourcePath);
    const targetPath = targetFor(sourcePath, title);
    if (targets.has(targetPath)) throw new Error(`musique_title_path_collision:folder_files:${targetPath}`);
    targets.add(targetPath);
    pathMap[sourcePath] = targetPath;
    articles.push({ source_path: sourcePath, target_path: targetPath, title,
      title_sha256: sha256(title), content_sha256: sha256(files[sourcePath]) });
  }
  if (!articles.length || sourceArticlePaths.length !== articles.length)
    throw new Error(`musique_title_path_no_articles:${record.source_ids?.[0]}`);
  if (sourceArticlePaths.some(path => !/^articles\/\d+\.md$/.test(path)))
    throw new Error(`musique_title_path_source_set_invalid:${record.source_ids?.[0]}`);

  const originalExpectedFiles = record.semantics.expected_files;
  const expectedFilePathMap = {};
  const expectedFileRecords = [];
  for (const [sourcePath, body] of Object.entries(originalExpectedFiles ?? {})) {
    if (!pathMap[sourcePath]) {
      if (sourcePath.startsWith('articles/')) throw new Error(`musique_title_path_expected_map_missing:${sourcePath}`);
      continue;
    }
    expectedFilePathMap[sourcePath] = pathMap[sourcePath];
    expectedFileRecords.push({ source_path: sourcePath, target_path: pathMap[sourcePath], content_sha256: sha256(body) });
  }

  const newFiles = remapFileKeys(files, pathMap, 'folder_files');
  const newExpectedFiles = remapFileKeys(originalExpectedFiles, pathMap, 'expected_files');
  const newTaskFiles = replaceExplicitPathStrings(record.semantics.files, pathMap);
  const newReference = remapReferenceCalls(record.curriculum.reference, pathMap);
  const newEvidencePathLists = {};
  for (const key of EVIDENCE_PATH_LISTS) {
    if (Array.isArray(record.curriculum.evidence?.[key]))
      newEvidencePathLists[key] = remapEvidencePathList(record.curriculum.evidence[key], pathMap);
  }
  const newRetrieved = newEvidencePathLists.retrieved ?? record.curriculum.evidence?.retrieved ?? [];

  record.semantics.folder_files = newFiles;
  if (originalExpectedFiles != null) record.semantics.expected_files = newExpectedFiles;
  record.semantics.files = newTaskFiles;
  record.curriculum.reference = newReference;
  if (record.curriculum.evidence) {
    for (const [key, values] of Object.entries(newEvidencePathLists)) record.curriculum.evidence[key] = values;
  }

  const review = { registry: REVISION, base_ir_id: record.id,
    source_id: record.external_source?.source_id ?? record.source_ids?.[0],
    source_snapshot_sha256: record.external_source?.snapshot_sha256 ?? null,
    path_map: pathMap, articles, expected_file_path_map: expectedFilePathMap,
    expected_file_records: expectedFileRecords, evidence_path_lists: newEvidencePathLists,
    retrieved_paths: newRetrieved };
  record.id = `${record.id}${VARIANT_SUFFIX}`;
  record.generation = { ...record.generation, article_path_review: review };
  validateVariant(record, review);
  return record;
}
