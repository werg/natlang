export type TerminalBlock =
  | { kind: 'text' | 'code' | 'status'; text: string; tone?: 'normal' | 'muted' | 'good' | 'warn' | 'bad' }
  | { kind: 'list'; items: string[]; ordered?: boolean }
  | { kind: 'table'; columns: string[]; rows: string[][] };
export type TerminalView = { title?: string; subtitle?: string; blocks: TerminalBlock[];
  prompt?: string; busy?: boolean; help?: string[] };

export type TerminalRenderOptions = { width?: number; color?: boolean };
const ansi = { reset: '\u001b[0m', bold: '\u001b[1m', dim: '\u001b[2m',
  muted: '\u001b[2m', good: '\u001b[32m', warn: '\u001b[33m', bad: '\u001b[31m' };

function clean(value: unknown): string {
  return String(value ?? '').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u009b]/g, '');
}
function styled(text: string, code: string, color: boolean): string {
  return color ? `${code}${text}${ansi.reset}` : text;
}
function widths(columns: string[], rows: string[][], width: number): number[] {
  const natural = columns.map((column, index) => Math.max(clean(column).length,
    ...rows.map(row => clean(row[index]).length)));
  const budget = Math.max(columns.length * 4, width - (columns.length - 1) * 3);
  while (natural.reduce((a, b) => a + b, 0) > budget) {
    const largest = Math.max(...natural), at = natural.indexOf(largest);
    if (natural[at]! <= 4) break;
    natural[at]!--;
  }
  return natural;
}
function cell(value: unknown, width: number): string {
  const text = clean(value).replace(/\s+/g, ' ');
  return (text.length > width ? `${text.slice(0, Math.max(1, width - 1))}…` : text).padEnd(width);
}

/** Render checked view data. Escape codes from view values are stripped. */
export function renderTerminalView(view: TerminalView, options: TerminalRenderOptions = {}): string {
  if (!view || !Array.isArray(view.blocks)) throw new TypeError('terminal view needs blocks');
  const width = Math.max(24, Math.floor(options.width ?? 80));
  const color = options.color ?? false;
  const lines: string[] = [];
  if (view.title) lines.push(styled(clean(view.title), ansi.bold, color));
  if (view.subtitle) lines.push(styled(clean(view.subtitle), ansi.dim, color));
  if (view.title || view.subtitle) lines.push('');
  for (const block of view.blocks) {
    if (!block || typeof block.kind !== 'string') throw new TypeError('invalid terminal view block');
    if (block.kind === 'text' || block.kind === 'code' || block.kind === 'status') {
      let text = clean(block.text);
      if (block.kind === 'code') text = text.split('\n').map(line => `  ${line}`).join('\n');
      const tone = block.tone && block.tone !== 'normal' ? ansi[block.tone] : '';
      lines.push(styled(text, tone, color));
    } else if (block.kind === 'list') {
      if (!Array.isArray(block.items)) throw new TypeError('terminal list needs items');
      block.items.forEach((item, index) => lines.push(`${block.ordered ? `${index + 1}.` : '•'} ${clean(item)}`));
    } else if (block.kind === 'table') {
      if (!Array.isArray(block.columns) || !Array.isArray(block.rows) ||
          block.rows.some(row => !Array.isArray(row) || row.length !== block.columns.length))
        throw new TypeError('terminal table rows must match columns');
      const sizes = widths(block.columns, block.rows, width);
      lines.push(block.columns.map((column, i) => cell(column, sizes[i]!)).join(' | '));
      lines.push(sizes.map(size => '-'.repeat(size)).join('-+-'));
      for (const row of block.rows) lines.push(row.map((value, i) => cell(value, sizes[i]!)).join(' | '));
    } else throw new TypeError(`unsupported terminal block: ${(block as { kind: string }).kind}`);
    lines.push('');
  }
  if (view.help?.length) lines.push(styled(view.help.map(clean).join('  '), ansi.dim, color), '');
  if (view.busy) lines.push(styled('Working…', ansi.warn, color));
  return lines.join('\n').replace(/\n+$/, '\n');
}
