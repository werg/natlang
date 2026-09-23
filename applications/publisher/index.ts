/**
 * Checked local Markdown/HTML publication. Natlang plans and composes a cited document from pinned
 * evidence passages; the publisher checks citations, tables and assets, renders both formats from the
 * same tree, and switches one symlink only after both files are written and the evidence is rechecked.
 */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { FolderHandle } from '@natlang/node';
import type { EvidenceCollection } from '../evidence/index.js';
import outline from './outline.nl';
import compose from './compose.nl';
import type { Document } from './types.js';

export type * from './types.js';
export type Table = { columns: string[], rows: (string | number)[][] };
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

/** Compose a cited document from pinned passages, then check and publish it. */
export async function publishBrief(publisher: DocumentPublisher, request: PublishRequest): Promise<PublishReport> {
  const { brief, span_ids, collection_revision, table_ids, asset_ids, target, files } = request;
  const passages = publisher.evidence.read(span_ids, collection_revision);
  const plan = await outline(brief, passages, table_ids, asset_ids, files);
  const document = await compose(brief, plan, passages, collection_revision, table_ids, asset_ids, files);
  const checked = publisher.check(document);
  return checked.ok ? publisher.publish(document, target) : rejected(target, checked.detail);
}
