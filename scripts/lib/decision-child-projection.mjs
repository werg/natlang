import { createHash } from 'node:crypto';
import ts from '../../ts-host/node_modules/typescript/lib/typescript.js';

// Project a completed inline result only when its declared type, exact
// callsite, and exact causal caller action establish one scalar output.
const sha = value => createHash('sha256').update(value).digest('hex');

function parse(code) {
  const source = ts.createSourceFile('decision-parent.ts', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const host = ts.createCompilerHost({ target: ts.ScriptTarget.Latest, noResolve: true, noLib: true });
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreateNewSourceFile) =>
    fileName === 'decision-parent.ts' ? source : getSourceFile(fileName, languageVersion, onError, shouldCreateNewSourceFile);
  host.fileExists = fileName => fileName === 'decision-parent.ts';
  host.readFile = fileName => fileName === 'decision-parent.ts' ? code : undefined;
  const program = ts.createProgram(['decision-parent.ts'], { target: ts.ScriptTarget.Latest, noResolve: true, noLib: true }, host);
  return { source, checker: program.getTypeChecker() };
}

function walk(node, visit) {
  visit(node);
  ts.forEachChild(node, child => walk(child, visit));
}

function symbolOf(checker, node) {
  return ts.isIdentifier(node) ? checker.getSymbolAtLocation(node) : undefined;
}

function nearestFunction(node) {
  for (let current = node.parent; current; current = current.parent)
    if (ts.isFunctionLike(current)) return current;
  return null;
}

function isConsumedExpression(node, shape) {
  let current = node;
  if (shape === 'decision') {
    if (!ts.isPropertyAccessExpression(current.parent) || current.parent.expression !== current || current.parent.name.text !== 'decision') return false;
    current = current.parent;
  } else if (shape === 'first') {
    if (!ts.isElementAccessExpression(current.parent) || current.parent.expression !== current ||
        !current.parent.argumentExpression || !ts.isNumericLiteral(current.parent.argumentExpression) ||
        current.parent.argumentExpression.text !== '0') return false;
    current = current.parent;
  } else if (ts.isPropertyAccessExpression(current.parent) && current.parent.expression === current ||
      ts.isElementAccessExpression(current.parent) && current.parent.expression === current) return false;
  const parent = current.parent;
  if (ts.isReturnStatement(parent) && parent.expression === current) return true;
  if (ts.isBinaryExpression(parent) && parent.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
      parent.right === current && ts.isElementAccessExpression(parent.left)) return true;
  // Permit only transparent literal packaging. Calls and operators can
  // transform the result and do not establish a source-label projection.
  for (let p = parent; p; p = p.parent) {
    if ((ts.isParenthesizedExpression(p) || ts.isAsExpression(p) || ts.isTypeAssertionExpression(p) ||
        ts.isSatisfiesExpression(p) || ts.isNonNullExpression(p)) && p.expression === current) { current = p; continue; }
    if (ts.isArrayLiteralExpression(p) && p.elements.includes(current)) { current = p; continue; }
    if (ts.isPropertyAssignment(p) && p.initializer === current) { current = p; continue; }
    if (ts.isObjectLiteralExpression(p) && p.properties.includes(current)) { current = p; continue; }
    if (ts.isReturnStatement(p)) return p.expression === current;
    return false;
  }
  return false;
}

function parentConsumer(code, childSite, childFunction, resultShape) {
  const { source, checker } = parse(code);
  let childDeclaration;
  walk(source, node => {
    if (!ts.isVariableDeclaration(node) || !ts.isIdentifier(node.name) || !node.initializer ||
        !(node.initializer.getStart(source) <= childSite.source_span.start && node.initializer.end >= childSite.source_span.end)) return;
    childDeclaration = node;
  });
  if (!childDeclaration) return null;
  const fnSymbol = symbolOf(checker, childDeclaration.name);
  if (!fnSymbol || fnSymbol.declarations?.length !== 1) return null;
  const awaitCalls = [];
  walk(source, node => {
    if (!ts.isVariableDeclaration(node) || !ts.isIdentifier(node.name) || !node.initializer ||
        !ts.isAwaitExpression(node.initializer) || !ts.isCallExpression(node.initializer.expression)) return;
    const expression = node.initializer.expression.expression;
    if (symbolOf(checker, expression) === fnSymbol) awaitCalls.push(node);
  });
  for (const call of awaitCalls) {
    const bindingSymbol = symbolOf(checker, call.name);
    if (!bindingSymbol || bindingSymbol.declarations?.length !== 1) continue;
    let mutatedAfterAwait = false;
    walk(source, node => {
      if (node.getStart(source) > call.end && isWriteOfSymbol(node, bindingSymbol, checker))
        mutatedAfterAwait = true;
    });
    if (mutatedAfterAwait) continue;
    let accepted = false;
    walk(source, node => {
      if (accepted || !ts.isIdentifier(node) || symbolOf(checker, node) !== bindingSymbol || node === call.name ||
          nearestFunction(node) !== nearestFunction(call)) return;
      if (isConsumedExpression(node, resultShape)) accepted = true;
    });
    if (accepted) return call.name.text;
  }
  return null;
}

function childFunctionAtSite(code, site) {
  const span = site?.source_span;
  if (!span || span.file !== 'eval' || !Number.isInteger(span.start) || !Number.isInteger(span.end) ||
      span.start < 0 || span.end > code.length || !code.slice(span.start, span.end).startsWith('nl.with')) return null;
  const { source, checker } = parse(code);
  let declaration;
  walk(source, node => {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer &&
        node.initializer.getStart(source) <= span.start && node.initializer.end >= span.end) declaration = node;
  });
  if (!declaration || symbolOf(checker, declaration.name)?.declarations?.length !== 1) return null;
  return declaration.name.text;
}

function awaitedResultBindings(code, functionName, site, shape) {
  if (!functionName) return [];
  const binding = parentConsumer(code, site, functionName, shape);
  return binding ? [binding] : [];
}

function identifierText(node, text) { return ts.isIdentifier(node) && node.text === text; }

function isWriteOfSymbol(node, symbol, checker) {
  if (!ts.isIdentifier(node) || symbolOf(checker, node) !== symbol) return false;
  const parent = node.parent;
  if (ts.isBinaryExpression(parent) && parent.left === node && parent.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
      parent.operatorToken.kind <= ts.SyntaxKind.LastAssignment) return true;
  if ((ts.isPrefixUnaryExpression(parent) || ts.isPostfixUnaryExpression(parent)) &&
      [ts.SyntaxKind.PlusPlusToken, ts.SyntaxKind.MinusMinusToken].includes(parent.operator)) return true;
  if (ts.isElementAccessExpression(parent) && parent.expression === node && ts.isBinaryExpression(parent.parent) &&
      parent.parent.left === parent && parent.parent.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
      parent.parent.operatorToken.kind <= ts.SyntaxKind.LastAssignment) return true;
  if (ts.isPropertyAccessExpression(parent) && parent.expression === node && ts.isBinaryExpression(parent.parent) &&
      parent.parent.left === parent && parent.parent.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
      parent.parent.operatorToken.kind <= ts.SyntaxKind.LastAssignment) return true;
  if (ts.isCallExpression(parent) && ts.isPropertyAccessExpression(parent.expression) && parent.expression.expression === node &&
      ['push', 'pop', 'splice', 'shift', 'unshift', 'sort', 'reverse', 'set', 'delete', 'clear'].includes(parent.expression.name.text)) return true;
  return false;
}

function outputMapIsReturnedOrRoundTripped(node, source, checker) {
  const parent = node.parent;
  if (ts.isReturnStatement(parent) && parent.expression === node) return true;
  if (!ts.isVariableDeclaration(parent) || parent.initializer !== node || !ts.isIdentifier(parent.name)) return false;
  const outputSymbol = symbolOf(checker, parent.name);
  if (!outputSymbol || outputSymbol.declarations?.length !== 1) return false;
  const variableStatement = parent.parent?.parent;
  const block = variableStatement?.parent;
  // Eval actions run at the source-file level. A return from an uncalled
  // nested helper does not establish that the trace-level output used this
  // projected map.
  if (!ts.isVariableStatement(variableStatement) || !ts.isSourceFile(block)) return false;
  const statements = block.statements;
  const mapStatementIndex = statements.indexOf(variableStatement);
  let mutated = false;
  walk(block, candidate => { if (isWriteOfSymbol(candidate, outputSymbol, checker)) mutated = true; });
  if (mutated) return false;
  const returnedDirectly = statements.slice(mapStatementIndex + 1).some(statement =>
    ts.isReturnStatement(statement) && ts.isIdentifier(statement.expression) &&
    symbolOf(checker, statement.expression) === outputSymbol);
  if (returnedDirectly) return true;
  const isFileMethod = (call, method) => {
    if (!ts.isCallExpression(call) || !ts.isPropertyAccessExpression(call.expression) || call.expression.name.text !== method) return null;
    const fileCall = call.expression.expression;
    if (!ts.isCallExpression(fileCall) || !ts.isPropertyAccessExpression(fileCall.expression) ||
        !identifierText(fileCall.expression.expression, 'folder') || fileCall.expression.name.text !== 'file') return null;
    return fileCall.getText(source);
  };
  const writeStatement = statements[mapStatementIndex + 1];
  const writeCall = ts.isExpressionStatement(writeStatement) && ts.isAwaitExpression(writeStatement.expression)
    ? writeStatement.expression.expression : null;
  if (!writeCall || !ts.isCallExpression(writeCall) || !ts.isPropertyAccessExpression(writeCall.expression) ||
      writeCall.expression.name.text !== 'writeJson' || writeCall.arguments.length !== 1 ||
      symbolOf(checker, writeCall.arguments[0]) !== outputSymbol) return false;
  const writePath = isFileMethod(writeCall, 'writeJson');
  const verifiedStatement = statements[mapStatementIndex + 2];
  if (!ts.isVariableStatement(verifiedStatement) || verifiedStatement.declarationList.declarations.length !== 1) return false;
  const verifiedDeclaration = verifiedStatement.declarationList.declarations[0];
  if (!ts.isIdentifier(verifiedDeclaration.name) || !verifiedDeclaration.initializer ||
      !ts.isAwaitExpression(verifiedDeclaration.initializer)) return false;
  const readCall = verifiedDeclaration.initializer.expression;
  if (!ts.isCallExpression(readCall) || !ts.isPropertyAccessExpression(readCall.expression) ||
      readCall.expression.name.text !== 'readJson' || isFileMethod(readCall, 'readJson') !== writePath) return false;
  const verifiedSymbol = symbolOf(checker, verifiedDeclaration.name);
  if (!verifiedSymbol || verifiedSymbol.declarations?.length !== 1) return false;
  let returned = false;
  walk(block, candidate => {
    if (!ts.isReturnStatement(candidate) || !ts.isIdentifier(candidate.expression) ||
        symbolOf(checker, candidate.expression) !== verifiedSymbol) return;
    if (statements.indexOf(candidate) > mapStatementIndex + 2) returned = true;
  });
  if (!returned) return false;
  return true;
}

function aggregateBoundToChild(code, site, childName) {
  const { source, checker } = parse(code);
  const aggregates = [];
  walk(source, node => {
    if (!ts.isVariableDeclaration(node) || !ts.isIdentifier(node.name) || !node.initializer ||
        !ts.isAwaitExpression(node.initializer) || !ts.isCallExpression(node.initializer.expression)) return;
    const promiseAll = node.initializer.expression;
    if (!ts.isPropertyAccessExpression(promiseAll.expression) || !identifierText(promiseAll.expression.expression, 'Promise') ||
        promiseAll.expression.name.text !== 'all') return;
    const declaration = childFunctionAtSite(code, site);
    if (!declaration) return;
    const parentDeclaration = (() => {
      let found;
      walk(source, candidate => {
        if (ts.isVariableDeclaration(candidate) && ts.isIdentifier(candidate.name) && candidate.name.text === declaration &&
            candidate.initializer?.getStart(source) <= site.source_span.start && candidate.initializer.end >= site.source_span.end) found = candidate;
      });
      return found;
    })();
    const childSymbol = parentDeclaration && symbolOf(checker, parentDeclaration.name);
    if (!childSymbol) return;
    let containsExactChildReturn = false;
    if (!promiseAll.arguments[0]) return;
    walk(promiseAll.arguments[0], candidate => {
      if (!ts.isReturnStatement(candidate) || !candidate.expression || !ts.isArrayLiteralExpression(candidate.expression) ||
          candidate.expression.elements.length !== 2) return;
      const pair = candidate.expression.elements;
      if (!ts.isIdentifier(pair[0]) || !ts.isAwaitExpression(pair[1]) || !ts.isCallExpression(pair[1].expression) ||
          symbolOf(checker, pair[1].expression.expression) !== childSymbol) return;
      containsExactChildReturn = true;
    });
    if (containsExactChildReturn) aggregates.push(node);
  });
  const aggregate = aggregates.length === 1 ? aggregates[0] : null;
  if (!aggregate || symbolOf(checker, aggregate.name)?.declarations?.length !== 1) return null;
  return aggregate.name.text;
}

function exactAggregateProjection(code, aggregateName) {
  const { source, checker } = parse(code);
  const refs = [];
  walk(source, node => {
    if (ts.isIdentifier(node) && node.text === aggregateName) refs.push(node);
  });
  if (refs.some(node => node.parent && (ts.isVariableDeclaration(node.parent) && node.parent.name === node ||
      ts.isBinaryExpression(node.parent) && node.parent.left === node ||
      ts.isCallExpression(node.parent) && ts.isPropertyAccessExpression(node.parent.expression) && node.parent.expression.expression === node &&
        ['push', 'pop', 'splice', 'shift', 'unshift', 'sort', 'reverse'].includes(node.parent.expression.name.text)))) return false;
  let projected = false;
  walk(source, node => {
    if (!ts.isCallExpression(node) || !ts.isPropertyAccessExpression(node.expression) || node.expression.name.text !== 'fromEntries' ||
        !identifierText(node.expression.expression, 'Object')) return;
    const map = node.arguments[0];
    if (!map || !ts.isCallExpression(map) || !ts.isPropertyAccessExpression(map.expression) || map.expression.name.text !== 'map' ||
        !identifierText(map.expression.expression, aggregateName)) return;
    const callback = map.arguments[0];
    if (!callback || !(ts.isArrowFunction(callback) || ts.isFunctionExpression(callback)) ||
        !ts.isArrayBindingPattern(callback.parameters[0]?.name) || callback.parameters[0].name.elements.length !== 2 ||
        !callback.body || !ts.isArrayLiteralExpression(callback.body)) return;
    const [id, answer] = callback.parameters[0].name.elements;
    const pair = callback.body.elements;
    if (!ts.isBindingElement(id) || !ts.isIdentifier(id.name) || !ts.isBindingElement(answer) || !ts.isIdentifier(answer.name) ||
        pair.length !== 2 || !identifierText(pair[0], id.name.text) || !ts.isElementAccessExpression(pair[1]) ||
        !identifierText(pair[1].expression, answer.name.text) || !pair[1].argumentExpression ||
        !ts.isNumericLiteral(pair[1].argumentExpression) || pair[1].argumentExpression.text !== '0') return;
    projected = outputMapIsReturnedOrRoundTripped(node, source, checker);
  });
  return projected;
}

/**
 * Return a scalar source decision only when the declared inline return type,
 * actual host value, and exact caller projection form one bound chain.
 */
export function projectDeclaredChildResult(returnType, value, parentCodes, inlineSite = null) {
  if (typeof returnType !== 'string' || (!Array.isArray(parentCodes) && typeof parentCodes !== 'string'))
    return { accepted: false, reason: 'missing-declared-type-or-parent-code' };
  const type = returnType.replace(/\s+/g, ' ').trim().replace(/\s*;\s*/g, ';');
  const codes = (Array.isArray(parentCodes) ? parentCodes : [parentCodes]).filter(code => typeof code === 'string');
  if (!codes.length) return { accepted: false, reason: 'exact-caller-action-not-bound' };
  const code = codes[0];
  const childFunction = childFunctionAtSite(code, inlineSite);
  if (!childFunction) return { accepted: false, reason: 'exact-child-callsite-not-bound' };

  if (type === 'string') {
    if (typeof value !== 'string') return { accepted: false, reason: 'declared-string-value-mismatch' };
    const binding = awaitedResultBindings(code, childFunction, inlineSite, 'string')[0];
    if (!binding) return { accepted: false, reason: 'exact-child-result-not-consumed' };
    return { accepted: true, value, projection: 'declared-string-return', result_binding: binding,
      projection_code_index: 0 };
  }

  if (type === 'string[]') {
    if (!Array.isArray(value) || value.length !== 1 || typeof value[0] !== 'string')
      return { accepted: false, reason: 'declared-string-array-value-mismatch' };
    const binding = awaitedResultBindings(code, childFunction, inlineSite, 'first')[0];
    if (binding) return { accepted: true, value: value[0], projection: 'declared-string-array-index-zero',
      result_binding: binding, projection_code_index: 0 };
    const aggregateName = aggregateBoundToChild(code, inlineSite, childFunction);
    if (aggregateName) {
      let invalidated = false;
      for (let index = 1; index < codes.length; index++) {
        const later = codes[index];
        const { source } = parse(later);
        walk(source, node => {
          if (invalidated || !ts.isIdentifier(node) || node.text !== aggregateName) return;
          const allowedMapReceiver = ts.isPropertyAccessExpression(node.parent) && node.parent.expression === node &&
            node.parent.name.text === 'map' && ts.isCallExpression(node.parent.parent) && node.parent.parent.expression === node.parent;
          if (!allowedMapReceiver) invalidated = true;
        });
        if (exactAggregateProjection(later, aggregateName) && !invalidated) return { accepted: true, value: value[0],
          projection: 'declared-string-array-index-zero-through-trace-bound-map',
          result_binding: `${aggregateName}.map(answer[0])`, projection_code_index: index,
          causal_aggregate: aggregateName, aggregate_source_action_index: 0 };
      }
    }
    return { accepted: false, reason: 'parent-does-not-project-exact-child-array-result' };
  }

  const hasDecisionString = /(?:^|[,{; ]|\breadonly\s+)decision\s*:\s*string\b/.test(type);
  if (hasDecisionString) {
    if (!value || typeof value !== 'object' || Array.isArray(value) || typeof value.decision !== 'string')
      return { accepted: false, reason: 'declared-decision-record-value-mismatch' };
    const binding = awaitedResultBindings(code, childFunction, inlineSite, 'decision')[0];
    if (!binding) return { accepted: false, reason: 'parent-does-not-project-exact-child-decision-field' };
    return { accepted: true, value: value.decision, projection: 'declared-record-decision-field',
      result_binding: binding, projection_code_index: 0 };
  }

  return { accepted: false, reason: 'unsupported-declared-return-type' };
}

/** Find exact eval code from the trajectory action which invoked this child. */
function parentEvalCodes(row, parentInvocationId) {
  if (typeof parentInvocationId !== 'string' || !Array.isArray(row?.trajectory)) return [];
  const codes = [];
  for (const step of row.trajectory) {
    if (step?.invocation_id !== parentInvocationId || step?.phase !== 'action') continue;
    for (const call of step.assistant?.calls ?? []) {
      if (call?.tool === 'eval' && typeof call.arguments?.code === 'string') codes.push(call.arguments.code);
    }
  }
  return codes;
}

export function parentEvalCode(row, parentInvocationId) {
  const codes = parentEvalCodes(row, parentInvocationId);
  return codes.length === 1 ? codes[0] : null;
}

/**
 * Follow only exact runtime callsite hashes. Later eval actions in the same
 * invocation cannot supply a projection for an earlier child result.
 */
export function parentEvalChain(row, ledger, childEntry) {
  if (!childEntry?.inline_instruction_site || !Array.isArray(ledger)) return [];
  const chain = [], seen = new Set();
  let child = childEntry;
  while (child && chain.length < 32) {
    const site = child.inline_instruction_site;
    const origin = site?.origin;
    const parentId = child.parent_invocation_id;
    if (!origin || typeof parentId !== 'string' || origin.parentInvocationId !== parentId ||
        typeof origin.writtenCodeSha256 !== 'string' || origin.writtenCodeSha256 !== origin.checkedCodeSha256 ||
        seen.has(parentId)) break;
    seen.add(parentId);
    const actionCodes = parentEvalCodes(row, parentId);
    const originIndex = actionCodes.findIndex(code => sha(code) === origin.writtenCodeSha256);
    if (originIndex < 0 || actionCodes.filter(code => sha(code) === origin.writtenCodeSha256).length !== 1) break;
    const parent = ledger.find(item => item?.invocation_id === parentId);
    chain.push({ invocation_id: parentId, code: actionCodes[originIndex],
      followup_codes: actionCodes.slice(originIndex + 1), entry: parent,
      action_sha256: origin.writtenCodeSha256, action_ordinal: origin.actionOrdinal, child_callsite: site });
    child = parent;
  }
  return chain;
}
