/** Source-backed directory tasks. Gold and reference metadata never enter the visible workspace. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join, posix } from 'node:path';
import { curriculumCase, evalCall, returnCall } from './lib.mjs';
import { buildBroaderSources } from './broader-sources.mjs';
import { reviewedMusiqueAliasFor } from './musique-reviewed-aliases.mjs';

export const SOURCE_ADAPTER_VERSION = 'natlang.directory_source_adapter/1';
export const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
export const permissive = license => ['MIT', 'APACHE-2.0', 'BSD-2-CLAUSE', 'BSD-3-CLAUSE', 'ISC', '0BSD'].includes(String(license).toUpperCase());
export function safePath(path) {
  if (typeof path !== 'string' || !path || path.includes('\\') || path.startsWith('/') || path.split('/').some(p => !p || p === '.' || p === '..'))
    throw new Error('unsafe_source_path');
  return posix.normalize(path);
}

export async function loadSourceCache(cache) {
  const manifest = JSON.parse(await readFile(join(cache, 'manifest.json'), 'utf8'));
  if (manifest.version !== 'natlang.directory_sources/1') throw new Error('unsupported_source_manifest');
  const sources = {};
  for (const [key, info] of Object.entries(manifest.sources)) {
    const raw = await readFile(join(cache, safePath(info.path)), 'utf8');
    if (digest(raw) !== info.sha256) throw new Error(`${key}: source_checksum_mismatch`);
    sources[key] = { info, rows: JSON.parse(raw) };
  }
  return { manifest, sources };
}

export function sourceCase({ source, info, sourceId, group, task, files, expectedFiles = files,
  expected, actions, oracle = 'exact', adaptation, license = info.license, treeInputs }) {
  const family = `source_${source}`, shape = digest([source, sourceId, SOURCE_ADAPTER_VERSION]).slice(0, 20);
  const record = curriculumCase({ family, shape, variant: 'v1', splitGroup: group,
    slice: 'folder_failure', domain: 'other', mode: 'single_call', inline: 'optional',
    root: treeInputs ? { name: 'update_tree', args: { state: 'Tree', utterance: 'string', history: 'unknown[]', system_acts: 'unknown[]' }, returns: 'Tree', instructions: task } :
      { name: 'process_workspace', kind: 'directory-reducer', args: {}, returns: 'string', instructions: task },
    ...(treeInputs ? { inputs: treeInputs, files: { 'types.ts': 'export type Tree = { name: string; children: Tree[] };' } } : { folderFiles: files, expectedFiles }),
    expected, split: 'train',
    reference: { root: actions }, evidence: { retrieved: Object.keys(files), world: [], background: [] } });
  record.source = source;
  record.task_modality = treeInputs ? 'tree-edit' : 'directory-reducer';
  record.source_ids = [sourceId];
  record.source_groups = [group];
  record.source_revisions = [info.revision ?? `snapshot:${info.sha256}`];
  record.license = license;
  record.gold_sources = [`${source}:original-gold`, 'native-file-replay'];
  record.semantics.oracle = oracle;
  record.generation = { generator: SOURCE_ADAPTER_VERSION, adaptation };
  record.external_source = { source, source_id: sourceId, original_split: info.original_split,
    revision: info.revision ?? null, snapshot_sha256: info.sha256, license,
    files: info.files, adaptation };
  return record;
}

const TATQA_SCALE_CONTRACT_REVISION = 'tatqa-evidence-scale-v2';
const TATQA_SCALE_CONTRACT = 'Use a nonempty scale only when the question or source evidence establishes it for the requested quantity. Use empty scale for dimensionless quantities or when no scale is stated. Do not infer scale from financial-report conventions or unrelated table rows.';

/** Add the evidence-bounded scale rule to future TATQA tasks and saved IR variants. */
export function applyEvidenceScaleContract(record) {
  if (record.source !== 'tatqa') return record;
  const suffix = ':evidence-scale-v2';
  const promptPath = record.semantics?.root;
  const prompt = record.semantics?.files?.[promptPath];
  if (typeof prompt !== 'string') throw new Error(`tatqa_scale_contract_prompt_missing:${record.source_ids?.[0]}`);

  if (record.id.endsWith(suffix)) {
    if (prompt.split(TATQA_SCALE_CONTRACT).length - 1 !== 1 ||
        record.generation?.task_contract_revision !== TATQA_SCALE_CONTRACT_REVISION ||
        record.external_source?.task_contract_revision !== TATQA_SCALE_CONTRACT_REVISION)
      throw new Error(`tatqa_scale_contract_variant_mismatch:${record.source_ids?.[0]}`);
    return record;
  }
  if (record.id.includes(suffix)) throw new Error(`tatqa_scale_contract_id_malformed:${record.source_ids?.[0]}`);

  const matches = prompt.split(TATQA_SCALE_CONTRACT).length - 1;
  if (matches > 1) throw new Error(`tatqa_scale_contract_prompt_duplicated:${record.source_ids?.[0]}`);
  let nextPrompt = prompt;
  if (matches === 0) {
    const insertion = 'Preserve source files.';
    if (!prompt.includes(insertion) || prompt.split(insertion).length !== 2)
      throw new Error(`tatqa_scale_contract_prompt_anchor_missing:${record.source_ids?.[0]}`);
    nextPrompt = prompt.replace(insertion, `${TATQA_SCALE_CONTRACT}\n${insertion}`);
  }

  record.semantics.files[promptPath] = nextPrompt;
  const baseId = record.id;
  record.id = `${baseId}${suffix}`;
  record.generation = { ...record.generation, task_contract_revision: TATQA_SCALE_CONTRACT_REVISION };
  record.external_source = { ...record.external_source, task_contract_revision: TATQA_SCALE_CONTRACT_REVISION };
  return record;
}

/** Reusable adapter for future builds and reviewed variants cloned from historical IR. */
export function applyReviewedMusiqueOracleAlias(record) {
  if (record.source !== 'musique') return record;
  const review = reviewedMusiqueAliasFor({
    sourceId: record.external_source?.source_id ?? record.source_ids?.[0],
    snapshotSha256: record.external_source?.snapshot_sha256,
    primary: record.semantics?.expected,
    files: record.semantics?.folder_files,
  });
  if (!review) return record;
  if (record.id.endsWith(':reviewed-alias-v1')) {
    if (!record.semantics.oracle?.alternates?.includes(review.accepted) ||
        record.generation?.oracle_review?.registry !== review.audit.registry)
      throw new Error(`reviewed_musique_alias_variant_contract_mismatch:${review.audit.source_id}`);
    return record;
  }

  const oracle = record.semantics.oracle;
  if (!oracle || typeof oracle !== 'object' || !Array.isArray(oracle.alternates))
    throw new Error(`reviewed_musique_alias_oracle_contract_missing:${review.audit.source_id}`);
  record.semantics.oracle = { ...oracle,
    alternates: [...new Set([...oracle.alternates, review.accepted])] };
  const audit = { ...review.audit, base_ir_id: record.id, variant: 'reviewed-alias-v1' };
  record.id = `${record.id}:reviewed-alias-v1`;
  record.generation = { ...record.generation, oracle_review: audit };
  record.external_source = { ...record.external_source, oracle_review: audit };
  return record;
}

// Whole-line replacements avoid constructing ambiguous partial-token edits from commits.
export function replacement(oldText, newText) {
  if (typeof oldText !== 'string' || typeof newText !== 'string' || oldText === newText) throw new Error('empty_edit');
  const oldLines = oldText.split('\n'), newLines = newText.split('\n');
  let first = 0;
  for (; first < Math.min(oldLines.length, newLines.length) && oldLines[first] === newLines[first]; first++) {}
  let suffix = 0;
  for (; suffix < oldLines.length - first && suffix < newLines.length - first &&
    oldLines.at(-1 - suffix) === newLines.at(-1 - suffix); suffix++) {}
  // Include a previous context line for insertions, and to make matching unique.
  first = Math.max(0, first - 1);
  const oldEnd = oldLines.length - suffix, newEnd = newLines.length - suffix;
  const find = oldLines.slice(first, oldEnd).join('\n'), replace_with = newLines.slice(first, newEnd).join('\n');
  if (!find || oldText.split(find).length !== 2 || oldText.replace(find, replace_with) !== newText)
    throw new Error('nonunique_or_nonlocal_edit');
  return { find, replace_with };
}

export function buildTaskSources(sources, limit = 12) {
  const records = [], rejected = [];
  const add = (source, row, build) => {
    try { const record = build(); if (record) records.push(record); }
    catch (error) { rejected.push({ source, id: row, reason: error.message }); }
  };
  if (sources.workbench) {
    const { info, rows: [data] } = sources.workbench;
    const seen = new Set();
    for (const [index, row] of data.tasks.entries()) {
      if (seen.size >= limit) break;
      if (row.base_template !== 'Delete my last email from {name}') continue;
      const targetId = /email\.delete_email\.func\(email_id="(\d+)"\)/.exec(row.outcome)?.[1];
      if (!targetId || seen.has(targetId)) continue;
      seen.add(targetId);
      add('workbench', index, () => {
        const target = data.emails.find(email => email.email_id === targetId);
        if (!target || target['inbox/outbox'] !== 'inbox') throw new Error('missing_email_gold');
        const sender = target['sender/recipient'];
        const senderName = sender.split('.')[0];
        const relevant = data.emails.filter(email => email['inbox/outbox'] === 'inbox' && email['sender/recipient'] === sender);
        const latest = [...relevant].sort((a, b) => b.sent_datetime.localeCompare(a.sent_datetime));
        if (latest[0]?.email_id !== targetId || latest[1]?.sent_datetime === latest[0].sent_datetime)
          throw new Error('source_latest_email_disagreement');
        if (!row.task.toLowerCase().includes(sender.split('.')[0].toLowerCase())) throw new Error('sender_not_in_task');
        const fixture = [...relevant, ...data.emails.filter(e => e['inbox/outbox'] === 'inbox' && e['sender/recipient'] !== sender).slice(0, 6)];
        const files = Object.fromEntries(fixture.map(e => [`inbox/${e.email_id}.json`, JSON.stringify({
          id: e.email_id, sender: e['sender/recipient'], subject: e.subject, sent_datetime: e.sent_datetime }) + '\n']));
        const expectedFiles = { ...files }; delete expectedFiles[`inbox/${targetId}.json`];
        const code = `const items = await folder.files('inbox/*.json');
const emails = await Promise.all(items.map(async file => JSON.parse(await file.readText())));
const matching = emails.filter(email => email.sender.split('.')[0] === ${JSON.stringify(senderName)}).sort((a,b) => b.sent_datetime.localeCompare(a.sent_datetime));
await folder.file('inbox/' + matching[0].id + '.json').remove();
return matching[0].id;`;
        return sourceCase({ source: 'workbench', info, sourceId: `email-task:${index}`, group: `workbench:email-workspace:${info.revision}`,
          task: `${row.task}. Each inbox JSON file represents one email. Delete its file and return its email ID. Preserve every other file.\nThe fixture contains every inbox email from the requested sender plus distractors; dates and subjects retain their source values.`,
          files, expectedFiles, expected: targetId, actions: [evalCall(code), returnCall(targetId)],
          adaptation: 'delete-latest-email; all matching records retained; body omitted because criterion uses sender/date' });
      });
    }
  }
  if (sources.commitpack) {
    const { info, rows } = sources.commitpack;
    const seen = new Set();
    for (const row of rows) {
      if (seen.size >= limit) break;
      add('commitpack', row.commit, () => {
        if (!permissive(row.license)) throw new Error('repository_license_ineligible');
        const path = safePath(row.old_file);
        if (path !== row.new_file) throw new Error('rename_not_supported');
        if (row.old_contents.length + row.new_contents.length > 18000) throw new Error('oversize_file');
        const edit = replacement(row.old_contents, row.new_contents);
        const fingerprint = digest([row.repos, row.commit, path]);
        if (seen.has(fingerprint)) return;
        seen.add(fingerprint);
        const request = JSON.stringify({ path, ...edit }, null, 2) + '\n';
        if (path === 'change-request.json') throw new Error('request_path_collision');
        const files = { [path]: row.old_contents, 'change-request.json': request };
        const record = sourceCase({ source: 'commitpack', info, sourceId: fingerprint, group: `commitpack:repo:${row.repos.split(',')[0]}`,
          task: `Apply the explicit change request in change-request.json to its target file. Preserve every other file and return the target path.\nSource commit description: ${row.subject}\nThe explicit replacement defines this scoped edit task.`,
          files, expectedFiles: { ...files, [path]: row.new_contents }, expected: path, license: `MIT dataset; ${row.license} repository`,
          actions: [['read_file', { path: 'change-request.json' }], ['read_file', { path }], ['edit_file', { path, ...edit }], returnCall(path)],
          adaptation: 'explicit-edit-request-from-before-after; does not claim original underspecified instruction uniquely determines patch' });
        record.external_source.commit = { commit: row.commit, repositories: row.repos, old_file: row.old_file,
          new_file: row.new_file, repository_license: row.license, original_row_sha256: digest(row) };
        record.source_groups.push(`repository:${row.repos.split(',')[0]}`);
        return record;
      });
    }
  }
  if (sources.tatqa) {
    const { info, rows } = sources.tatqa;
    let count = 0;
    for (const context of rows) {
      if (count >= limit) break;
      // One question per context for diversity; labels and derivations stay out of the workspace.
      const question = context.questions.find(q => q.answer_type === 'arithmetic') ?? context.questions[0];
      add('tatqa', question.uid, () => {
        const answer = String(Array.isArray(question.answer) ? question.answer.join('; ') : question.answer);
        const files = { 'table.json': JSON.stringify(context.table.table, null, 2) + '\n',
          ...Object.fromEntries(context.paragraphs.map(p => [`notes/${p.uid}.md`, p.text + '\n'])) };
        let computation = JSON.stringify(answer);
        if (question.answer_type === 'arithmetic') {
          const expression = question.derivation;
          if (!expression || !/^[\d\s.+*/()%-]+$/.test(expression)) throw new Error('unsupported_derivation');
          const computed = Function(`"use strict"; return (${expression});`)();
          if (!Number.isFinite(computed) || Math.abs(computed - Number(question.answer)) > 0.011)
            throw new Error('derivation_answer_disagreement');
          computation = `String(Number((${expression}).toFixed(2)))`;
          if (String(Number(computed.toFixed(2))) !== answer) throw new Error('numeric_display_disagreement');
        }
        const expected = JSON.stringify({ answer, scale: question.scale ?? '' });
        const code = `const result = JSON.stringify({answer: ${computation}, scale: ${JSON.stringify(question.scale ?? '')}});
await folder.file('answer.json').writeText(result + '\\n');
return result;`;
        count++;
        const record = sourceCase({ source: 'tatqa', info, sourceId: question.uid, group: `tatqa:context:${context.table.uid}`,
          task: `${question.question}\nUse table.json and notes/. Write answer.json and return the same JSON string with exactly answer (a string; multiple spans separated by "; ") and scale ("", "percent", "thousand", "million" or "billion"). Preserve source files.`,
          files, expectedFiles: { ...files, 'answer.json': expected + '\n' }, expected,
          oracle: { level: 'normalized', normalization: 'json-string-record' },
          actions: [['read_file', { path: 'table.json' }], ...Object.keys(files).filter(path => path.startsWith('notes/'))
            .map(path => ['read_file', { path }]), evalCall(code), returnCall(expected)],
          adaptation: `read-evidence-before-calculation; original-${question.answer_type}-gold; explicit-display-format` });
        record.semantics.files_oracle = { compare: 'json-string-record', threshold: 1 };
        const marker = context.table.table.flat().find(cell => typeof cell === 'string' && cell.length > 12 && !question.question.includes(cell));
        if (marker) {
          record.curriculum.mode = 'followup';
          record.curriculum.decisive = [{ marker, source: 'file', note: 'Read the source table before choosing the computation.' }];
          record.curriculum.plausible_actions = ['read source evidence', 'return an unverified answer'];
        }
        applyEvidenceScaleContract(record);
        return record;
      });
    }
  }
  if (sources.musique) {
    const { info, rows } = sources.musique;
    const heldSeeds = new Set(info.held_out_seed_ids ?? []);
    let count = 0;
    for (const row of rows) {
      if (count >= limit) break;
      add('musique', row.id, () => {
        if (!row.answerable) throw new Error('unanswerable_requires_blocked_oracle');
        if (row.question_decomposition?.some(q => heldSeeds.has(String(q.id)))) throw new Error('held_out_musique_seed');
        if (!row.paragraphs?.length || !row.answer || !row.paragraphs.some(p => p.is_supporting)) throw new Error('missing_support');
        const files = Object.fromEntries(row.paragraphs.map(p => [`articles/${p.idx}.md`, `# ${p.title}\n\n${p.paragraph_text}\n`]));
        if (Object.values(files).join('').length > 22000) throw new Error('oversize_context');
        count++;
        const record = sourceCase({ source: 'musique', info, sourceId: row.id,
          // Shared paragraphs connect many tasks; keeping this pilot together is conservative.
          group: `musique:train-v1.0-pilot`,
          task: `${row.question}\nRead the articles to answer. Return only the answer text. Preserve the workspace.`,
          files, expected: row.answer, actions: [evalCall(`const files = await folder.files('articles/*.md');\nfor (const file of files) await file.readText();`), returnCall(row.answer)],
          // Preserve exact source-backed aliases. Token overlap alone cannot establish
          // equivalence, and partial span agreement is held by the training quality gate.
          oracle: { level: 'normalized', alternates: row.answer_aliases ?? [] },
          adaptation: 'original-train-question; all supplied paragraphs; no generated evidence' });
        applyReviewedMusiqueOracleAlias(record);
        record.source_groups.push(...(row.question_decomposition ?? []).map(q => `musique:seed:${q.id}`));
        record.source_groups.push(...row.paragraphs.map(p => `document:${digest([p.title, p.paragraph_text])}`));
        return record;
      });
    }
  }
  const broader = buildBroaderSources(sources, limit, sourceCase);
  return { records: [...records, ...broader.records], rejected: [...rejected, ...broader.rejected] };
}
