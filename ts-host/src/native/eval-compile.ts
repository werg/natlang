/**
 * TypeScript to JavaScript for eval code, shared by every platform. A platform contributes only a realm to run the
 * result in (native/realm.ts); nothing here depends on one.
 */
import ts from 'typescript';
import { PRELUDE } from './prelude.generated.js';

export type EvalCompileOptions = {
  /** The code is the body of a function (it may `return`) rather than a program whose completion value is its result. */
  body: boolean;
  /** The body may `await` (an async function). */
  asyncBody?: boolean;
  /**
   * Package imports are lowered to `__natlang_import(specifier)` calls (static and dynamic); this validates each
   * specifier and throws for one the eval may not import. Without it imports are left as written and fail to run.
   */
  imports?: (specifier: string) => void;
  /**
   * `await x` becomes `(await __natlang_bindAwait(x))()`, which restores the natlang task after the await. For platforms
   * whose context store cannot follow promises (browsers); the realm must bind `__natlang_bindAwait`.
   */
  restoreAfterAwait?: boolean;
};

/** The name a compiled body function has; the realm body calls it. */
const BODY_FUNCTION = '__natlang_body';

function lowerStaticImports(source: string, validateImport: (specifier: string) => void): string {
  const file = ts.createSourceFile('imports.ts', source, ts.ScriptTarget.Latest, true);
  const edits: { start: number; end: number; text: string }[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isImportDeclaration(node)) {
      if (!ts.isStringLiteral(node.moduleSpecifier)) throw new Error('Package import must use a string literal');
      validateImport(node.moduleSpecifier.text);
      const clause = node.importClause;
      const call = `await __natlang_import(${node.moduleSpecifier.getText(file)})`;
      const fields: string[] = [];
      let text = '';
      const typeOnlyBindings = clause?.namedBindings && ts.isNamedImports(clause.namedBindings)
        && clause.namedBindings.elements.length > 0 && clause.namedBindings.elements.every(binding => binding.isTypeOnly);
      if (!clause?.isTypeOnly && !(typeOnlyBindings && !clause?.name)) {
        if (clause?.name) fields.push(`default: ${clause.name.text}`);
        const bindings = clause?.namedBindings;
        if (bindings && ts.isNamedImports(bindings)) for (const binding of bindings.elements)
          if (!binding.isTypeOnly) fields.push(`${binding.propertyName?.text ?? binding.name.text}: ${binding.name.text}`);
        if (fields.length) text += `const { ${fields.join(', ')} } = ${call};`;
        if (bindings && ts.isNamespaceImport(bindings)) text += `const ${bindings.name.text} = ${call};`;
        if (!text) text = `${call};`;
      }
      edits.push({ start: node.getStart(file), end: node.end, text });
    } else ts.forEachChild(node, visit);
  };
  visit(file);
  for (const edit of edits.sort((a, b) => b.start - a.start)) source = source.slice(0, edit.start) + edit.text + source.slice(edit.end);
  return source;
}

/** Browsers have no async context: restore the natlang task frame after every await in eval code. */
const restoreAfterAwait: ts.TransformerFactory<ts.SourceFile> = context => file => {
  const f = context.factory;
  const visit: ts.Visitor = node => ts.isAwaitExpression(node) ?
    f.createCallExpression(f.createParenthesizedExpression(f.createAwaitExpression(f.createCallExpression(
      f.createIdentifier('__natlang_bindAwait'), undefined, [ts.visitNode(node.expression, visit) as ts.Expression]))), undefined, []) :
    ts.visitEachChild(node, visit, context);
  return ts.visitNode(file, visit) as ts.SourceFile;
};

const lowerImportCalls: ts.TransformerFactory<ts.SourceFile> = context => root => {
  const visit: ts.Visitor = node => {
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword)
      return ts.factory.updateCallExpression(node, ts.factory.createIdentifier('__natlang_import'), undefined,
        node.arguments.map(arg => ts.visitNode(arg, visit) as ts.Expression));
    return ts.visitEachChild(node, visit, context);
  };
  return ts.visitNode(root, visit) as ts.SourceFile;
};

/** The JavaScript for `code`. A body is a function declaration; call `evalRealmBody` for what a realm runs. */
export function compileEval(code: string, options: EvalCompileOptions): string {
  let source = options.body ? `${options.asyncBody ? 'async ' : ''}function ${BODY_FUNCTION}() {\n${code}\n}` : code;
  // Reparse lowered imports before TS binding; otherwise TS rewrites uses to removed import aliases.
  if (options.imports) source = lowerStaticImports(source, options.imports);
  const result = ts.transpileModule(source, {
    fileName: 'natlang-eval.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None, isolatedModules: true, removeComments: false },
    transformers: { before: [...(options.imports ? [lowerImportCalls] : []), ...(options.restoreAfterAwait ? [restoreAfterAwait] : [])] },
  });
  const errors = (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new SyntaxError('TypeScript: ' + errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('; '));
  return result.outputText;
}

/**
 * The strict-mode function body a realm runs for compiled eval code: the prelude, the code, and its result (the body
 * function's, or the program's completion value). The names a realm binds around it are `self`, `__live`,
 * `__natlang_import`, `console` and, with `restoreAfterAwait`, `__natlang_bindAwait` (see `EVAL_PARAMETERS`).
 */
export function evalRealmBody(compiled: string, body: boolean): string {
  return body ? `${PRELUDE}\n${compiled}\nreturn ${BODY_FUNCTION}();` : `${PRELUDE}\nreturn eval(${JSON.stringify(`"use strict";\n${compiled}`)});`;
}
