/** Checked local Markdown/HTML publication from a portable document tree. */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

const sha = text => createHash('sha256').update(text).digest('hex');
const escapeHtml = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const escapeMd = value => String(value).replace(/([\\`*_[\]<>|])/g, '\\$1');

export class DocumentPublisher {
  constructor(root, { evidence, tables = {}, assets = {} } = {}) {
    this.root = resolve(root);
    this.evidence = evidence;
    this.tables = structuredClone(tables);
    this.assets = structuredClone(assets);
    this.events = [];
  }

  check(document) {
    try {
      if (!document || !document.title || !Array.isArray(document.sections) ||
          !document.sections.length || !Array.isArray(document.assets))
        throw new Error('document lacks a title or sections');
      if (!this.evidence || document.evidence_revision !== this.evidence.revision())
        throw new Error('evidence revision changed');
      for (const section of document.sections) {
        if (!section.heading || typeof section.body !== 'string' || !Array.isArray(section.claims))
          throw new Error('invalid section');
        if (section.table_id && !Object.hasOwn(this.tables, section.table_id))
          throw new Error(`unknown table: ${section.table_id}`);
        if (section.table_id) {
          const table = this.tables[section.table_id];
          if (!Array.isArray(table.columns) || !table.columns.length ||
              !Array.isArray(table.rows) || table.rows.some(row =>
                !Array.isArray(row) || row.length !== table.columns.length))
            throw new Error(`invalid table: ${section.table_id}`);
        }
      }
      for (const asset of document.assets)
        if (!Object.hasOwn(this.assets, asset) ||
            typeof this.assets[asset] !== 'string' ||
            !/^(?:https:\/\/|\.\/|\/)/.test(this.assets[asset]))
          throw new Error(`unknown or unsafe asset: ${asset}`);
      const claims = document.sections.flatMap(section => section.claims);
      const ids = [...new Set(claims.map(claim => claim.span_id))];
      const passages = this.evidence.read(ids, document.evidence_revision);
      const checked = this.evidence.verify(passages,
        { answer: '', claims, gaps: [] }, document.evidence_revision);
      if (checked.status === 'invalid-citation') throw new Error(checked.detail);
      return { ok: true, revision: sha(JSON.stringify(document)), detail: '' };
    } catch (error) {
      return { ok: false, revision: '', detail: error instanceof Error ? error.message : String(error) };
    }
  }

  renderText(document) {
    const markdown = [`# ${escapeMd(document.title)}`, ''];
    const html = ['<!doctype html>', '<meta charset="utf-8">',
      `<title>${escapeHtml(document.title)}</title>`, `<h1>${escapeHtml(document.title)}</h1>`];
    for (const section of document.sections) {
      markdown.push(`## ${escapeMd(section.heading)}`, '', escapeMd(section.body), '');
      html.push(`<section><h2>${escapeHtml(section.heading)}</h2><p>${escapeHtml(section.body).replace(/\n/g, '<br>')}</p>`);
      if (section.table_id) {
        const table = this.tables[section.table_id];
        const columns = table.columns, rows = table.rows;
        markdown.push(`| ${columns.map(escapeMd).join(' | ')} |`,
          `| ${columns.map(() => '---').join(' | ')} |`,
          ...rows.map(row => `| ${row.map(escapeMd).join(' | ')} |`), '');
        html.push('<table><thead><tr>', ...columns.map(cell => `<th>${escapeHtml(cell)}</th>`),
          '</tr></thead><tbody>', ...rows.map(row => '<tr>' +
            row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('') + '</tr>'), '</tbody></table>');
      }
      for (const claim of section.claims) {
        markdown.push(`> ${escapeMd(claim.text)} [${escapeMd(claim.span_id)} @ ${escapeMd(claim.revision.slice(0, 12))}]`, '');
        html.push(`<blockquote>${escapeHtml(claim.text)} <cite data-span="${escapeHtml(claim.span_id)}"` +
          ` data-revision="${escapeHtml(claim.revision)}">${escapeHtml(claim.quote)}</cite></blockquote>`);
      }
      html.push('</section>');
    }
    for (const id of document.assets) {
      const src = this.assets[id];
      markdown.push(`![${escapeMd(id)}](${encodeURI(src)})`, '');
      html.push(`<img alt="${escapeHtml(id)}" src="${escapeHtml(src)}">`);
    }
    return { markdown: markdown.join('\n'), html: html.join('\n') + '\n' };
  }

  async publish(document, target) {
    if (!/^[a-z][a-z0-9_-]*$/.test(target))
      return { status: 'rejected', target, revision: '', markdown_sha256: '', html_sha256: '', detail: 'invalid target' };
    let snapshot;
    try { snapshot = structuredClone(document); }
    catch { return { status: 'rejected', target, revision: '', markdown_sha256: '', html_sha256: '', detail: 'document is not portable' }; }
    const checked = this.check(snapshot);
    if (!checked.ok) return { status: 'rejected', target, revision: '',
      markdown_sha256: '', html_sha256: '', detail: checked.detail };
    const texts = this.renderText(snapshot);
    const version = `${target}-${randomUUID()}`;
    const relative = join('.versions', version);
    const directory = join(this.root, relative);
    const temporaryLink = join(this.root, `.${version}.link`);
    let linked = false;
    try {
      await mkdir(directory, { recursive: true });
      await writeFile(join(directory, 'document.md'), texts.markdown);
      await writeFile(join(directory, 'document.html'), texts.html);
      const again = this.check(snapshot);
      if (!again.ok || again.revision !== checked.revision)
        return { status: 'rejected', target, revision: '', markdown_sha256: '', html_sha256: '',
          detail: again.detail || 'document changed during rendering' };
      await symlink(relative, temporaryLink, 'dir');
      const finalCheck = this.check(snapshot);
      if (!finalCheck.ok || finalCheck.revision !== checked.revision)
        return { status: 'rejected', target, revision: '', markdown_sha256: '', html_sha256: '',
          detail: finalCheck.detail || 'document changed before publication' };
      await rename(temporaryLink, join(this.root, target));
      linked = true;
    } finally {
      if (!linked) {
        await rm(temporaryLink, { force: true });
        await rm(directory, { recursive: true, force: true });
      }
    }
    const report = { status: 'prepared', target, revision: checked.revision,
      markdown_sha256: sha(await readFile(join(this.root, target, 'document.md'), 'utf8')),
      html_sha256: sha(await readFile(join(this.root, target, 'document.html'), 'utf8')), detail: '' };
    this.events.push({ operation: 'publisher.prepare', target,
      revision: report.revision, markdown_sha256: report.markdown_sha256,
      html_sha256: report.html_sha256 });
    return report;
  }

  drainEvents() { return this.events.splice(0); }
}
