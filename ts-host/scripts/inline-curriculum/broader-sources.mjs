/** Author-annotated scientific evidence and non-file tree transitions. No model calls. */
import { createHash } from 'node:crypto';
import { evalCall, returnCall } from './lib.mjs';
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
          return sourceCase({ source: 'qasper', info, sourceId: `${paper.id}:${qa.question_id}`, group: `qasper:paper:${paper.id}`,
            task: `${qa.question}\nRead the paper sections. Return the exact answer span(s), sorted lexicographically and joined with a newline. Preserve all files.`,
            files, expected: spans.join('\n'), actions: [evalCall("const files = await folder.files('paper/*.md');\nfor (const file of files) await file.readText();"), returnCall(spans.join('\n'))],
            adaptation: 'original train extractive QA; full textual paper; unanimous span annotations; visual/free-form/unanswerable cases held' });
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
          const edits = treeEdits(turn.input_dialog_state, turn.target_dialog_state);
          if (!edits.length || edits.length > 3) throw new Error('unbounded_or_no_tree_change');
          // Only changes with explicit new leaf values in the current utterance qualify for this pilot.
          if (!edits.some(edit => edit.values.length) || edits.flatMap(edit => edit.values).some(value =>
            !norm(turn.utterance).includes(norm(value)))) throw new Error('implicit_tree_value_requires_review');
          const inputs = { state: turn.input_dialog_state, utterance: turn.utterance, history: structuredClone(history), system_acts: turn.input_system_acts ?? [] };
          if (JSON.stringify(inputs).length > 16000) throw new Error('oversize_tree_context');
          count++;
          const record = sourceCase({ source: 'treedst', info, sourceId: turn.turn_id, group: `treedst:conversation:${conversation.session_id}`,
            task: 'Update the existing dialogue goal tree to reflect the current utterance in its conversation context and the preceding system acts. Keep the name/children schema, preserve unaffected branches, and return the updated Tree. Children are keyed by name: sibling order does not change meaning, and duplicate child names are invalid. Work on a mutable copy of the input tree. This is a tree edit, not a request to perform the real-world booking or action.',
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
