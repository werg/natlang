import ts from 'typescript';
import { hexDigest } from '../native/hash.js';

type Dict = Record<string, unknown>;
type Span = { file?: unknown; start: number; end: number; line: number; column: number };

function isDict(value: unknown): value is Dict { return !!value && typeof value === 'object' && !Array.isArray(value); }
function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (isDict(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`;
  return JSON.stringify(value) ?? 'undefined';
}
function spanAt(source: ts.SourceFile, node: ts.Node): Span {
  const start = node.getStart(source), end = node.getEnd();
  const position = source.getLineAndCharacterOfPosition(start);
  return { file: 'eval', start, end, line: position.line + 1, column: position.character + 1 };
}
function templateSegments(template: ts.TemplateLiteral): { strings: string[]; expressions: string[] } {
  if (ts.isNoSubstitutionTemplateLiteral(template)) return { strings: [template.text], expressions: [] };
  return { strings: [template.head.text, ...template.templateSpans.map(item => item.literal.text)],
    expressions: template.templateSpans.map(item => item.expression.getText(template.getSourceFile())) };
}
function matchingTag(node: ts.TaggedTemplateExpression, site: Dict): boolean {
  const explicit = site.explicit_captures === true;
  const call = ts.isCallExpression(node.tag) ? node.tag : undefined;
  const withTag = call && ts.isPropertyAccessExpression(call.expression) &&
    ts.isIdentifier(call.expression.expression) && call.expression.expression.text === 'nl' &&
    call.expression.name.text === 'with';
  if (!!withTag !== explicit) return false;
  if (explicit) {
    if (!call || call.arguments.length !== 1 || !ts.isObjectLiteralExpression(call.arguments[0]!)) return false;
    const captureNames = call.arguments[0]!.properties.map(property =>
      ts.isShorthandPropertyAssignment(property) ? property.name.text :
        ts.isPropertyAssignment(property) && (ts.isIdentifier(property.name) || ts.isStringLiteral(property.name)) ?
          property.name.text : undefined);
    const declared = Array.isArray(site.captures) ? site.captures.map(item => isDict(item) ? item.name : undefined) : [];
    return captureNames.every(name => typeof name === 'string') && stable(captureNames) === stable(declared);
  }
  return ts.isIdentifier(node.tag) && node.tag.text === 'nl';
}

/**
 * Map a compiler plan's helper-prefixed coordinates into its exact action source.
 * The mapping only succeeds for one TypeScript AST site whose cooked segments,
 * interpolation expressions, tag form and captures agree with the compiler plan.
 */
export function deriveInlineActionSpanMap(code: string, site: Dict): Dict | undefined {
  const origin = isDict(site.origin) ? site.origin : undefined;
  const writtenHash = origin?.writtenCodeSha256;
  const checkedHash = origin?.checkedCodeSha256;
  if (!origin || typeof writtenHash !== 'string' || !/^[a-f0-9]{64}$/.test(writtenHash) ||
      writtenHash !== hexDigest(code) || checkedHash !== writtenHash || site.soft_body_id !== undefined) return undefined;
  const oldSource = isDict(site.source_span) ? site.source_span : undefined;
  const oldTemplate = isDict(site.template_span) ? site.template_span : undefined;
  const oldChecked = isDict(site.checked_template_span) ? site.checked_template_span : undefined;
  if (!oldSource || !oldTemplate || !oldChecked || !Array.isArray(site.template_segments) ||
      !Array.isArray(site.interpolations)) return undefined;
  const source = ts.createSourceFile('inline-action.ts', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const segments = site.template_segments;
  const interpolations = site.interpolations;
  const matches: { source: Span; template: Span; interpolationSpans: Span[] }[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isTaggedTemplateExpression(node) && matchingTag(node, site)) {
      const parts = templateSegments(node.template);
      const expressions = interpolations.map(item => isDict(item) ? item.expression : undefined);
      if (stable(parts.strings) === stable(segments) && stable(parts.expressions) === stable(expressions)) {
        const sourceSpan = spanAt(source, node), templateSpan = spanAt(source, node.template);
        const interpolationSpans = ts.isNoSubstitutionTemplateLiteral(node.template) ? [] :
          node.template.templateSpans.map(item => spanAt(source, item.expression));
        matches.push({ source: sourceSpan, template: templateSpan, interpolationSpans });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  if (matches.length !== 1) return undefined;
  const found = matches[0]!;
  const integer = (value: unknown): value is number => Number.isSafeInteger(value);
  const sourceDelta = Number(oldSource.start) - found.source.start;
  const templateDelta = Number(oldTemplate.start) - found.template.start;
  const checkedDelta = Number(oldChecked.start) - found.template.start;
  if (![oldSource.start, oldSource.end, oldSource.line, oldSource.column,
        oldTemplate.start, oldTemplate.end, oldTemplate.line, oldTemplate.column,
        oldChecked.start, oldChecked.end].every(integer) ||
      sourceDelta <= 0 || sourceDelta !== templateDelta || sourceDelta !== checkedDelta ||
      Number(oldSource.end) - found.source.end !== sourceDelta ||
      Number(oldTemplate.end) - found.template.end !== sourceDelta ||
      Number(oldChecked.end) - found.template.end !== sourceDelta ||
      Number(oldSource.column) !== found.source.column || Number(oldTemplate.column) !== found.template.column ||
      Number(oldSource.line) - found.source.line !== Number(oldTemplate.line) - found.template.line)
    return undefined;
  const mappedChecked = { ...found.template };
  return { schema: 'natlang.inline-action-span-map/1', method: 'unique-exact-typescript-ast-site',
    action_source_sha256: writtenHash, prefix_characters: sourceDelta,
    original: { source_span: oldSource, template_span: oldTemplate, checked_template_span: oldChecked },
    mapped: { source_span: found.source, template_span: found.template, checked_template_span: mappedChecked,
      interpolation_spans: found.interpolationSpans } };
}

/** Verify a materializer-produced span map by repeating the AST and exact-hash binding. */
export function verifyInlineActionSpanMap(code: string, site: Dict): boolean {
  const map = isDict(site.action_local_span_map) ? site.action_local_span_map : undefined;
  if (!map || map.schema !== 'natlang.inline-action-span-map/1') return false;
  const original = isDict(map.original) ? map.original : undefined;
  const mapped = isDict(map.mapped) ? map.mapped : undefined;
  if (!original || !mapped) return false;
  const candidate = { ...site, source_span: original.source_span, template_span: original.template_span,
    checked_template_span: original.checked_template_span };
  const recomputed = deriveInlineActionSpanMap(code, candidate);
  if (!recomputed || stable(recomputed) !== stable(map)) return false;
  return stable(site.source_span) === stable(mapped.source_span) &&
    stable(site.template_span) === stable(mapped.template_span) &&
    stable(site.checked_template_span) === stable(mapped.checked_template_span);
}
