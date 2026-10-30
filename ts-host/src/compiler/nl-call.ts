import ts from 'typescript';

/**
 * `nl` called like a function on its instructions, nl(`...`) or nl<T>(`...`), is the one-shot call it reads as:
 * nl`...`(), a new natural-language call with the interpolated values, whose result is the answer. Rewritten in
 * place with the same length, so positions in diagnostics still point into the code as written. Instructions that
 * are not a literal (a string built at run time) stay as they are, and are reported.
 */
export function desugarNlCalls(source: string): string {
  if (!/\bnl\s*(?:<[^()]*>)?\s*\(/.test(source)) return source;
  return applyNlCallEdits(source, nlCallEdits(ts.createSourceFile('snippet.ts', source, ts.ScriptTarget.ES2022, true)));
}

export type NlCallEdit = { start: number; end: number; text: string };

/**
 * The same-length rewrites `desugarNlCalls` makes in a parsed file. `accept` decides, for each call whose callee is
 * spelled `nl`, whether that name is the built-in (a host that resolves symbols passes its checker's answer, so a
 * local `nl` stays an ordinary function); by default every such call is.
 */
export function nlCallEdits(file: ts.SourceFile, accept: (callee: ts.Identifier) => boolean = () => true): NlCallEdit[] {
  const source = file.text;
  const edits: NlCallEdit[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'nl' &&
        node.arguments.length === 1 && accept(node.expression)) {
      const argument = node.arguments[0]!;
      let template: string | undefined;
      if (ts.isNoSubstitutionTemplateLiteral(argument) || ts.isTemplateExpression(argument)) template = argument.getText(file);
      else if (ts.isStringLiteral(argument) && !/[`\\]|\$\{/.test(argument.text)) template = `\`${argument.text}\``;
      const open = node.arguments.pos - 1;  // the "(" before the argument
      if (template !== undefined && template.length === argument.getText(file).length && source[open] === '(') {
        edits.push({ start: open, end: node.getEnd(), text: `${template}()` });
        return;
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return edits;
}

export const applyNlCallEdits = (source: string, edits: readonly NlCallEdit[]): string =>
  [...edits].reverse().reduce((text, edit) => text.slice(0, edit.start) + edit.text + text.slice(edit.end), source);
