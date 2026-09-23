import ts from 'typescript';

/** Add body level capture to synchronous declarations. The caller owns execution and the output file. */
export function instrumentSource(source, metadata = {}) {
  const file = ts.createSourceFile(metadata.path ?? 'capture.ts', source, ts.ScriptTarget.Latest, true,
    /\.tsx?$/.test(metadata.path ?? '') ? ts.ScriptKind.TS : ts.ScriptKind.JS);
  const edits = [];
  const functions = [];
  for (const node of file.statements) {
    if (!ts.isFunctionDeclaration(node) || !node.name || !node.body) continue;
    let skipReason;
    if (node.modifiers?.some((m) => m.kind === ts.SyntaxKind.AsyncKeyword) || node.asteriskToken) skipReason = 'async/generator declarations are deferred';
    else if (node.parameters.some((p) => !ts.isIdentifier(p.name))) skipReason = 'destructured parameters are deferred';
    else {
      let collision = false;
      const findGeneratedName = (n) => {
        if (ts.isIdentifier(n) && n.text.startsWith('__cc_')) collision = true;
        ts.forEachChild(n, findGeneratedName);
      };
      findGeneratedName(node);
      if (collision) skipReason = 'reserved instrumentation identifier __cc_* is already used';
    }
    if (skipReason) { functions.push({ name: node.name.text, instrumented: false, reason: skipReason }); continue; }
    const names = node.parameters.map((p) => p.name.text);
    const key = `${metadata.sourceName ?? 'source'}:${metadata.revision ?? 'unknown'}:${metadata.path ?? 'capture.ts'}:${node.name.text}`;
    const id = JSON.stringify({ key, source: metadata.sourceName ?? 'unknown', revision: metadata.revision ?? 'unknown', path: metadata.path ?? 'capture.ts', function: node.name.text });
    const cap = `globalThis.__codeCorpusCapture`;
    const argExpr = `[${names.join(', ')}]`;
    const entry = `\nconst __cc_id = ${id};\nconst __cc_args = ${argExpr};\n${cap}.enter(__cc_id, __cc_args);\ntry {`;
    // Rewrite only returns owned by this declaration, preserving nested callback returns.
    const returns = [];
    const walk = (n) => {
      if (n !== node.body && (ts.isFunctionLike(n))) return;
      if (ts.isReturnStatement(n)) returns.push(n);
      ts.forEachChild(n, walk);
    };
    walk(node.body);
    for (const ret of returns) {
      const expr = ret.expression?.getText(file) ?? 'undefined';
      const value = `__cc_result_${ret.pos}`;
      edits.push({ start: ret.getStart(file), end: ret.end, text: `{ const ${value} = (${expr}); ${cap}.returned(__cc_id, ${value}); return ${value}; }` });
    }
    const bodyStart = node.body.getStart(file);
    const bodyEnd = node.body.end - 1;
    const close = `\n${cap}.returned(__cc_id, undefined);\n} catch (__cc_error) { ${cap}.thrown(__cc_id, __cc_error); throw __cc_error; } finally { ${cap}.finish(__cc_id, __cc_args); }\n`;
    if (bodyEnd === bodyStart + 1) edits.push({ start: bodyEnd, end: bodyEnd, text: `${entry}${close}` });
    else {
      edits.push({ start: bodyStart + 1, end: bodyStart + 1, text: entry });
      edits.push({ start: bodyEnd, end: bodyEnd, text: close });
    }
    functions.push({ name: node.name.text, key, instrumented: true });
  }
  let output = source;
  for (const e of edits.sort((a, b) => b.start - a.start || b.end - a.end)) output = output.slice(0, e.start) + e.text + output.slice(e.end);
  if (/\.tsx?$/.test(metadata.path ?? '')) output = ts.transpileModule(output, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
  return { code: output, functions };
}
