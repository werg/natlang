/**
 * Checked local Markdown/HTML publication. Natlang plans and composes a cited document from pinned
 * evidence passages; the publisher checks citations, tables and assets, renders both formats from the
 * same tree, and switches one symlink only after both files are written and the evidence is rechecked.
 */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { untrusted, type FolderHandle, type Untrusted } from '@natlang/node';
import type { EvidenceCollection } from '../evidence/index.js';
import outline from './outline.nl';
import readNote from './read_note.nl';
import writeSection from './write_section.nl';
import { outlineProblems, unsourcedNumbers } from './checks.js';
import type { Document, OutlineSection, Passage, Section, Table } from './types.js';

export type * from './types.js';
export * from './checks.js';
export type PublishCheck = { ok: boolean, revision: string, detail: string };
export type PublishReport = { status: 'prepared' | 'rejected', target: string, revision: string,
  markdown_sha256: string, html_sha256: string, detail: string };
export type PublishRequest = { brief: string, span_ids: string[], collection_revision: string, table_ids: string[],
  asset_ids: string[], target: string, files?: FolderHandle };

const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const escapeHtml = (value: unknown) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const escapeMd = (value: unknown) => String(value).replace(/([\\`*_[\]<>|])/g, '\\$1');
const rejected = (target: string, detail: string): PublishReport =>
  ({ status: 'rejected', target, revision: '', markdown_sha256: '', html_sha256: '', detail });

export class DocumentPublisher {
  readonly root: string;
  private readonly tables: Record<string, Table>;
  private readonly assets: Record<string, string>;
  private readonly events: Record<string, unknown>[] = [];

  constructor(root: string, readonly evidence: EvidenceCollection,
    { tables = {}, assets = {} }: { tables?: Record<string, Table>, assets?: Record<string, string> } = {}) {
    this.root = resolve(root);
    this.tables = structuredClone(tables);
    this.assets = structuredClone(assets);
  }

  /** A table of the store, or null when the id names none. */
  table(id: string): Table | null { return this.tables[id] ? structuredClone(this.tables[id]!) : null; }

  check(document: Document): PublishCheck {
    try {
      if (!document || !document.title || !Array.isArray(document.sections) || !document.sections.length || !Array.isArray(document.assets))
        throw new Error('document lacks a title or sections');
      if (document.evidence_revision !== this.evidence.revision()) throw new Error('evidence revision changed');
      for (const section of document.sections) {
        if (!section.heading || typeof section.body !== 'string' || !Array.isArray(section.claims)) throw new Error('invalid section');
        if (!section.table_id) continue;
        const table = this.tables[section.table_id];
        if (!table) throw new Error(`unknown table: ${section.table_id}`);
        if (!Array.isArray(table.columns) || !table.columns.length || !Array.isArray(table.rows) ||
            table.rows.some(row => !Array.isArray(row) || row.length !== table.columns.length))
          throw new Error(`invalid table: ${section.table_id}`);
      }
      for (const asset of document.assets)
        if (typeof this.assets[asset] !== 'string' || !/^(?:https:\/\/|\.\/|\/)/.test(this.assets[asset]!))
          throw new Error(`unknown or unsafe asset: ${asset}`);
      const claims = document.sections.flatMap(section => section.claims);
      const passages = this.evidence.read([...new Set(claims.map(claim => claim.span_id))], document.evidence_revision);
      const checked = this.evidence.verify(passages, { answer: '', claims, gaps: [] }, document.evidence_revision);
      if (checked.status === 'invalid-citation') throw new Error(checked.detail);
      return { ok: true, revision: sha(JSON.stringify(document)), detail: '' };
    } catch (error) {
      return { ok: false, revision: '', detail: error instanceof Error ? error.message : String(error) };
    }
  }

  renderText(document: Document): { markdown: string, html: string } {
    const markdown = [`# ${escapeMd(document.title)}`, ''];
    const html = ['<!doctype html>', '<meta charset="utf-8">', `<title>${escapeHtml(document.title)}</title>`, `<h1>${escapeHtml(document.title)}</h1>`];
    for (const section of document.sections) {
      markdown.push(`## ${escapeMd(section.heading)}`, '', escapeMd(section.body), '');
      html.push(`<section><h2>${escapeHtml(section.heading)}</h2><p>${escapeHtml(section.body).replace(/\n/g, '<br>')}</p>`);
      if (section.table_id) {
        const { columns, rows } = this.tables[section.table_id]!;
        markdown.push(`| ${columns.map(escapeMd).join(' | ')} |`, `| ${columns.map(() => '---').join(' | ')} |`,
          ...rows.map(row => `| ${row.map(escapeMd).join(' | ')} |`), '');
        html.push('<table><thead><tr>', ...columns.map(cell => `<th>${escapeHtml(cell)}</th>`), '</tr></thead><tbody>',
          ...rows.map(row => '<tr>' + row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('') + '</tr>'), '</tbody></table>');
      }
      for (const claim of section.claims) {
        markdown.push(`> ${escapeMd(claim.text)} [${escapeMd(claim.span_id)} @ ${escapeMd(claim.revision.slice(0, 12))}]`, '');
        html.push(`<blockquote>${escapeHtml(claim.text)} <cite data-span="${escapeHtml(claim.span_id)}"` +
          ` data-revision="${escapeHtml(claim.revision)}">${escapeHtml(claim.quote)}</cite></blockquote>`);
      }
      html.push('</section>');
    }
    for (const id of document.assets) {
      const src = this.assets[id]!;
      markdown.push(`![${escapeMd(id)}](${encodeURI(src)})`, '');
      html.push(`<img alt="${escapeHtml(id)}" src="${escapeHtml(src)}">`);
    }
    return { markdown: markdown.join('\n'), html: html.join('\n') + '\n' };
  }

  /** Render and publish atomically under `root/target`, a symlink to a fresh version directory. */
  async publish(document: Document, target: string): Promise<PublishReport> {
    if (!/^[a-z][a-z0-9_-]*$/.test(target)) return rejected(target, 'invalid target');
    let snapshot: Document;
    try { snapshot = structuredClone(document); } catch { return rejected(target, 'document is not portable'); }
    const checked = this.check(snapshot);
    if (!checked.ok) return rejected(target, checked.detail);
    const texts = this.renderText(snapshot);
    const version = `${target}-${randomUUID()}`, relative = join('.versions', version);
    const directory = join(this.root, relative), temporaryLink = join(this.root, `.${version}.link`);
    let linked = false;
    try {
      await mkdir(directory, { recursive: true });
      await writeFile(join(directory, 'document.md'), texts.markdown);
      await writeFile(join(directory, 'document.html'), texts.html);
      if (sha(await readFile(join(directory, 'document.md'), 'utf8')) !== sha(texts.markdown) ||
          sha(await readFile(join(directory, 'document.html'), 'utf8')) !== sha(texts.html))
        throw new Error('rendered file verification failed');
      const again = this.check(snapshot);
      if (!again.ok || again.revision !== checked.revision) return rejected(target, again.detail || 'document changed during rendering');
      await symlink(relative, temporaryLink, 'dir');
      const final = this.check(snapshot);
      if (!final.ok || final.revision !== checked.revision) return rejected(target, final.detail || 'document changed before publication');
      await rename(temporaryLink, join(this.root, target));
      linked = true;
    } finally {
      if (!linked) { await rm(temporaryLink, { force: true }); await rm(directory, { recursive: true, force: true }); }
    }
    const report: PublishReport = { status: 'prepared', target, revision: checked.revision,
      markdown_sha256: sha(texts.markdown), html_sha256: sha(texts.html), detail: '' };
    this.events.push({ operation: 'publisher.prepare', target, revision: report.revision,
      markdown_sha256: report.markdown_sha256, html_sha256: report.html_sha256 });
    return report;
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

type Written = { section: Section } | { problem: string };

/**
 * One section: the model writes body and claims over the section's passages; the host fills revisions and checks the
 * citations and the numbers. A section that fails is written once more with the problems; the others are untouched.
 */
async function composeSection(publisher: DocumentPublisher, brief: string, planned: OutlineSection, passages: Passage[],
    table: Table | null, note: Untrusted<string>): Promise<Written> {
  let problem = '';
  for (let attempt = 0; attempt < 2; attempt++) {
    const draft = problem ? await writeSection(brief, planned.heading, planned.purpose, passages, table, note, problem) :
      await writeSection(brief, planned.heading, planned.purpose, passages, table, note);
    const claims = publisher.evidence.cite(draft.claims, passages);
    const found = [...publisher.evidence.citationProblems(passages, claims).map(row => row.problem)];
    const numbers = unsourcedNumbers(draft.body, passages, table);
    if (numbers.length) found.push(`the body states numbers that neither the passages nor the table give: ${numbers.join(', ')}`);
    if (!found.length) return { section: { heading: planned.heading, body: draft.body, claims, ...(planned.table_id ? { table_id: planned.table_id } : {}) } };
    problem = found.join('\n');
  }
  return { problem };
}

/**
 * Compose a cited document from pinned passages, then check and publish it. The outline assigns passages, a table and
 * assets to each section; the sections are written at once; the host adds the evidence revision, the claim revisions,
 * the table and the assets, and the exact check and atomic publish decide the rest.
 */
export async function publishBrief(publisher: DocumentPublisher, request: PublishRequest): Promise<PublishReport> {
  const { brief, span_ids, collection_revision, table_ids, asset_ids, target, files } = request;
  const passages = publisher.evidence.read(span_ids, collection_revision);
  const note = untrusted(files ? await readNote(brief, files) : '', 'editorial note');
  const offered = { passage_ids: passages.map(passage => passage.id), table_ids, asset_ids };
  let plan = await outline(brief, passages, table_ids, asset_ids, note);
  let problems = outlineProblems(plan, offered);
  if (problems.length) {
    plan = await outline(brief, passages, table_ids, asset_ids, note, problems.join('\n'));
    problems = outlineProblems(plan, offered);
  }
  if (problems.length) return rejected(target, `invalid outline: ${problems.join('; ')}`);
  const byId = new Map(passages.map(passage => [passage.id, passage]));
  const written = await Promise.all(plan.sections.map(planned => composeSection(publisher, brief, planned,
    [...new Set(planned.passage_ids)].map(id => byId.get(id)!), planned.table_id ? publisher.table(planned.table_id) : null, note)));
  const failed = written.flatMap((result, index) => 'problem' in result ?
    [`section ${JSON.stringify(plan.sections[index]!.heading)}: ${result.problem.split('\n').join('; ')}`] : []);
  if (failed.length) return rejected(target, failed.join(' | '));
  const document: Document = { title: plan.title, evidence_revision: collection_revision,
    sections: written.map(result => (result as { section: Section }).section),
    assets: [...new Set(plan.sections.flatMap(planned => planned.asset_ids))] };
  const checked = publisher.check(document);
  return checked.ok ? publisher.publish(document, target) : rejected(target, checked.detail);
}
