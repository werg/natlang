/** Author-annotated scientific evidence and non-file tree transitions. No model calls. */
import { createHash } from 'node:crypto';
import { evalCall, returnCall } from './lib.mjs';
import { hasExistingTreeValueContract } from '../../dist/teacher/tree-contract.js';
import { namedTreeCanonical } from '../../dist/teacher/oracle.js';
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const norm = value => value.toLowerCase().replace(/\s+/g, ' ').trim();

export function treeEdits(before, after, path = 'state') {
  const valid = node => node && typeof node.name === 'string' && Array.isArray(node.children) &&
    Object.keys(node).every(key => ['name', 'children'].includes(key));
  if (!valid(before) || !valid(after)) throw new Error('invalid_tree_schema');
  if (same(before, after)) return [];
  if (!before.children.length && !after.children.length) {
    return [{ code: `${path}.name = ${JSON.stringify(after.name)};`, values: [after.name] }];
  }
  if (before.name !== after.name) throw new Error('tree_domain_or_operator_change');
  const edits = [];
  const unique = children => new Set(children.map(c => c.name)).size === children.length;
  if (!unique(before.children) || !unique(after.children)) throw new Error('ambiguous_tree_siblings');
  // Match structural children by name, preserving existing branches and order. A one-child value
  // replacement is a leaf edit, while slot additions/deletions are explicit splice operations.
  if (before.children.length === 1 && after.children.length === 1) {
    return treeEdits(before.children[0], after.children[0], `${path}.children[0]`);
  }
  let working = structuredClone(before.children);
  for (let i = working.length - 1; i >= 0; i--) {
    if (!after.children.some(child => child.name === working[i].name)) {
      edits.push({ code: `${path}.children.splice(${i}, 1);`, values: [] });
      working.splice(i, 1);
    }
  }
  for (let i = 0; i < after.children.length; i++) {
    const target = after.children[i];
    const found = working.findIndex(child => child.name === target.name);
    if (found < 0) {
      const values = [];
      const collect = node => { if (!valid(node)) throw new Error('invalid_tree_schema');
        if (!node.children.length) values.push(node.name); else node.children.forEach(collect); };
      collect(target);
      edits.push({ code: `${path}.children.splice(${i}, 0, ${JSON.stringify(target)});`, values });
      working.splice(i, 0, target);
    } else {
      if (found !== i) throw new Error('tree_reordering_not_supported');
      edits.push(...treeEdits(working[i], target, `${path}.children[${i}]`));
    }
  }
  return edits;
}

export function buildBroaderSources(sources, limit, sourceCase) {
  const records = [], rejected = [];
  const attempt = (source, id, build) => { try { const row = build(); if (row) records.push(row); }
    catch (error) { rejected.push({ source, id, reason: error.message }); } };
  if (sources.qasper) {
    const { info, rows } = sources.qasper;
    const held = new Set(info.held_out_ids ?? []);
    let count = 0;
    for (const paper of rows) {
      if (count >= limit) break;
      for (const qa of paper.qas ?? []) {
        if (count >= limit) break;
        attempt('qasper', qa.question_id, () => {
          if (held.has(paper.id)) throw new Error('held_out_paper');
          const annotations = (qa.answers ?? []).map(item => item.answer);
          if (!annotations.length || annotations.some(a => !a || a.unanswerable || !a.extractive_spans?.length ||
            a.free_form_answer || a.evidence?.some(e => e.startsWith('FLOAT SELECTED'))))
            throw new Error('nonextractive_or_visual_answer');
          const spans = [...annotations[0].extractive_spans].sort();
          if (annotations.some(a => !same([...a.extractive_spans].sort(), spans))) throw new Error('answer_annotation_disagreement');
          const files = {};
          for (const [s, section] of (paper.full_text ?? []).entries()) {
            files[`paper/section-${s}.md`] = `# ${section.section_name}\n\n${section.paragraphs.join('\n\n')}\n`;
          }
          files['paper/abstract.md'] = `# ${paper.title}\n\n${paper.abstract}\n`;
          const text = Object.values(files).join('');
          if (text.length > 36000) throw new Error('oversize_paper');
          if (spans.some(span => !span || !text.includes(span)) || annotations.some(a =>
            !a.evidence?.length || a.evidence.some(e => !text.includes(e)))) throw new Error('missing_exact_text_evidence');
          count++;
          const sourceId = `${paper.id}:${qa.question_id}`;
          const evidence = qasperVisibleEvidence(annotations, files, spans, sourceId, info);
          const record = sourceCase({ source: 'qasper', info, sourceId: `${paper.id}:${qa.question_id}`, group: `qasper:paper:${paper.id}`,
            task: `${qa.question}\nUse the cited paper excerpts to answer. Return the exact answer span(s), sorted lexicographically and joined with a newline. Preserve all files.`,
            files, expected: spans.join('\n'), actions: evidence.actions,
            adaptation: 'original train extractive QA; full textual paper; unanimous span annotations; visual/free-form/unanswerable cases held' });
          record.id += ':evidence-visible-v6';
          record.curriculum.variant = 'qasper-evidence-visible-v6';
          record.curriculum.family_version = Number(record.curriculum.family_version ?? 1) + 1;
          record.generation.evidence_visibility_revision = 'qasper-human-evidence-cited-source-excerpts-v6';
          record.external_source.evidence_visibility_revision = 'qasper-human-evidence-cited-source-excerpts-v6';
          const visibilityNote = 'source-annotation-directed cited paper excerpts are displayed before the scripted answer (reference construction, not model-generated reasoning)';
          record.generation.adaptation += `; original QASPER evidence/source excerpts shown in bounded native eval reads; ${visibilityNote}`;
          record.external_source.evidence_annotation_source = evidence.provenance;
          return record;
        });
      }
    }
  }
  if (sources.scifact) {
    const { info, rows: [data] } = sources.scifact;
    const corpus = new Map(data.corpus.map(doc => [String(doc.doc_id), doc]));
    const held = new Set(info.held_out_ids ?? []);
    let count = 0;
    for (const claim of data.claims) {
      if (count >= limit) break;
      attempt('scifact', claim.id, () => {
        const docs = Object.keys(claim.evidence ?? {});
        if (docs.length !== 1) throw new Error('missing_or_multiple_evidence_documents');
        if (docs.some(id => held.has(id)) || claim.cited_doc_ids.some(id => held.has(String(id)))) throw new Error('held_out_scientific_document');
        const annotations = claim.evidence[docs[0]];
        if (!annotations.length || annotations.some(a => !['SUPPORT', 'CONTRADICT'].includes(a.label)) ||
          new Set(annotations.map(a => a.label)).size !== 1) throw new Error('claim_label_disagreement');
        const doc = corpus.get(docs[0]);
        const evidence = [...new Set(annotations.flatMap(a => a.sentences))].sort((a, b) => a - b);
        if (!doc || !evidence.length || evidence.some(i => !Number.isInteger(i) || !doc.abstract[i])) throw new Error('missing_claim_evidence');
        const files = Object.fromEntries([...new Set(claim.cited_doc_ids.map(String))].map(id => {
          const article = corpus.get(id); if (!article) throw new Error('missing_cited_document');
          return [`abstracts/${id}.json`, JSON.stringify({ id, title: article.title, sentences: article.abstract }) + '\n'];
        }));
        if (!files[`abstracts/${docs[0]}.json`]) throw new Error('evidence_not_in_cited_documents');
        // Different annotators can select different sufficient sentence sets. Do not require their
        // union as the only valid explanation; preserve annotations outside the model's opening.
        const expected = annotations[0].label;
        count++;
        const record = sourceCase({ source: 'scifact', info, sourceId: `claim:${claim.id}`, group: `scifact:document:${docs[0]}`,
          task: `Assess this claim against the supplied abstracts: ${claim.claim}\nRead the relevant documents and return only SUPPORT or CONTRADICT. Use only these documents and preserve all files.`,
          files, expected, actions: [evalCall("const files = await folder.files('abstracts/*.json');\nfor (const file of files) await file.readText();"), returnCall(expected)],
          adaptation: 'train claim/evidence annotations; document-linked dev exclusion; no medical advice or generated facts' });
        record.source_groups.push(...claim.cited_doc_ids.map(id => `scifact:document:${id}`));
        record.external_source.evidence_annotations = claim.evidence;
        return record;
      });
    }
  }
  if (sources.treedst) {
    const { info, rows } = sources.treedst;
    let count = 0;
    for (const conversation of rows) {
      if (count >= limit) break;
      const history = [];
      for (const turn of conversation.turns ?? []) {
        if (count >= limit) break;
        attempt('treedst', turn.turn_id, () => {
          if (!turn.input_dialog_state || !turn.target_dialog_state) throw new Error('not_an_existing_tree_edit');
          if (namedTreeCanonical(turn.input_dialog_state) === null || namedTreeCanonical(turn.target_dialog_state) === null)
            throw new Error('invalid_or_duplicate_named_children');
          if (!hasExistingTreeValueContract(turn.input_dialog_state, turn.target_dialog_state))
            throw new Error('unverified_tree_transition_contract');
          const edits = treeEdits(turn.input_dialog_state, turn.target_dialog_state);
          if (!edits.length || edits.length > 3) throw new Error('unbounded_or_no_tree_change');
          // Only changes with explicit new leaf values in the current utterance qualify for this pilot.
          if (!edits.some(edit => edit.values.length) || edits.flatMap(edit => edit.values).some(value =>
            !norm(turn.utterance).includes(norm(value)))) throw new Error('implicit_tree_value_requires_review');
          const inputs = { state: turn.input_dialog_state, utterance: turn.utterance, history: structuredClone(history), system_acts: turn.input_system_acts ?? [] };
          if (JSON.stringify(inputs).length > 16000) throw new Error('oversize_tree_context');
          count++;
          const record = sourceCase({ source: 'treedst', info, sourceId: turn.turn_id, group: `treedst:conversation:${conversation.session_id}`,
            task: 'Update the existing dialogue goal tree to reflect the current utterance in its conversation context and the preceding system acts. Keep the name/children schema, preserve unaffected branches, and return the updated Tree. Children are keyed by name: sibling order does not change meaning, and duplicate child names are invalid. Keep every existing structural branch and comparator; change only the explicitly requested literal values in existing slots. Work on a mutable copy of the input tree. This is a tree edit, not a request to perform the real-world booking or action.',
            files: {}, treeInputs: inputs, expected: turn.target_dialog_state,
            oracle: { level: 'normalized', normalization: 'named-tree' },
            actions: [evalCall('const updated = JSON.parse(JSON.stringify(state));\n' +
              edits.map(edit => edit.code.replace(/^state/, 'updated')).join('\n') + '\nreturn updated;'), ['reply', { text: 'done' }]],
            adaptation: 'original train existing-tree transition; bounded structural patch; explicit new leaf values; full preceding context; no file-tree wrapper' });
          record.source_groups.push(`treedst:input:${hash(turn.input_dialog_state)}`);
          return record;
        });
        history.push({ utterance: turn.utterance, system_acts: turn.input_system_acts ?? [], goal: turn.target_dialog_state });
      }
    }
  }
  return { records, rejected };
}

function qasperVisibleEvidence(annotations, files, spans, sourceId, info) {
  const selected = [];
  const add = item => {
    if (!selected.some(x => x.path === item.path && x.start === item.start && x.end === item.end)) selected.push(item);
  };
  const norm = text => text.replace(/\s+/g, ' ').trim();
  const paragraphRanges = (path, text) => {
    const ranges = [];
    let offset = 0;
    for (const paragraph of text.split('\n\n')) {
      const length = paragraph.length;
      ranges.push({ path, start: offset, end: offset + length, text: paragraph });
      offset += length + 2;
    }
    return ranges;
  };
  const first = annotations[0];
  for (const evidence of first.evidence ?? []) {
    if (!evidence || evidence.startsWith('FLOAT SELECTED')) throw new Error('non_text_qasper_evidence');
    const hits = [];
    for (const [path, text] of Object.entries(files)) {
      if (!path.startsWith('paper/section-')) continue;
      const exact = text.indexOf(evidence);
      if (exact >= 0) hits.push({ path, start: exact, end: exact + evidence.length, kind: 'human-annotation-evidence' });
      else for (const p of paragraphRanges(path, text)) {
        if (norm(p.text).includes(norm(evidence))) hits.push({ path, start: p.start, end: p.end, kind: 'human-annotation-evidence-paragraph' });
      }
    }
    if (!hits.length) throw new Error('qasper_evidence_not_in_source_files');
    add(hits[0]);
  }
  const hasSpan = span => selected.some(x => files[x.path].slice(x.start, x.end).includes(span));
  for (const span of [...new Set(annotations[0].extractive_spans ?? [])]) {
    if (hasSpan(span)) continue;
    const hit = Object.entries(files).filter(([path]) => path.startsWith('paper/section-'))
      .flatMap(([path, text]) => paragraphRanges(path, text).filter(p => p.text.includes(span))
        .map(p => ({ path, start: p.start, end: p.end, kind: 'gold-span-source-paragraph' })))[0];
    if (!hit) throw new Error('qasper_answer_span_not_in_source_files');
    add(hit);
  }
  if ([...new Set(annotations[0].extractive_spans ?? [])].some(span => !hasSpan(span)))
    throw new Error('qasper_source_evidence_incomplete');
  const actions = [];
  for (const range of selected) {
    const chunks = [];
    for (let start = range.start; start < range.end;) {
      let end = Math.min(start + 600, range.end);
      if (end < range.end) {
        const last = files[range.path].charCodeAt(end - 1);
        if (last >= 0xd800 && last <= 0xdbff) end--;
      }
      chunks.push({ start, end });
      start = end;
    }
    for (const [index, { start, end }] of chunks.entries()) {
      const total = chunks.length;
      const part = index + 1;
      const code = `await (async () => {\n` +
        `  const text = await folder.file(${JSON.stringify(range.path)}).readText();\n` +
        `  console.log('Citation: ${range.path} [UTF-16 offsets ${start}-${end}, excerpt ${part}/${total}]\\n' + text.slice(${start}, ${end}));\n` +
        `})();`;
      actions.push(evalCall(code));
    }
  }
  const source_file_sha256 = Object.fromEntries([...new Set(selected.map(x => x.path))].map(path =>
    [path, createHash('sha256').update(files[path]).digest('hex')]));
  const [paper_id, question_id] = sourceId.split(':', 2);
  const provenance = { archive_sha256: info.files?.find(file => file.sha256)?.sha256 ?? null,
    member: 'qasper-train-v0.3.json', paper_id, question_id, annotation_count: annotations.length,
    selected_answer_annotation: 0, source_ranges: selected, source_file_sha256,
    source_paragraph_gold_span_backfills: selected.filter(x => x.kind === 'gold-span-source-paragraph').length,
    reference_eval_code_sha256: actions.map(([, args]) => createHash('sha256').update(args.code).digest('hex')),
    selection_method: 'first original extractive annotation evidence; full containing original paragraph on source-whitespace fallback; for any gold span not present in those excerpts, add its full original source paragraph; runtime outputs only cited source text; reference action selection is annotation-directed and is not represented as model-generated reasoning' };
  return { actions: [...actions, returnCall(spans.join('\n'))], provenance };
}
