// Extract source evidence for human semantic review; never use candidate roles as admission policy.
const fs = require("node:fs"),
  path = require("node:path"),
  crypto = require("node:crypto");
const ts = require("../../ts-host/node_modules/typescript");
const args = process.argv.slice(2);
if (args.length !== 2) {
  console.error(
    "Usage: node scripts/neuralese_data/inventory_semantic_checks.cjs TEMPLATE_WORKTREE_ROOT OUTPUT.json",
  );
  process.exit(2);
}

const r = path.resolve(args[0]);
const roots = [
  "system-testing",
  "personal",
  "spectrolite",
  "news",
  "examples",
  "google-workspace",
  "system",
  "base",
].filter((x) => fs.existsSync(path.join(r, x)));
const entries = JSON.parse(
  fs.readFileSync(path.join(r, "base/skills/open-user-goal/entries.json")),
);
const known = new Set(Object.keys(entries));
const sources = new Map();
const checks = [];
function walk(dir) {
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .flatMap((e) =>
      e.name === "node_modules" || e.name === ".git"
        ? []
        : e.isDirectory()
          ? walk(path.join(dir, e.name))
          : e.name.endsWith(".ts") && !e.name.includes(".test.")
            ? [path.join(dir, e.name)]
            : [],
    );
}
for (const root of roots)
  for (const file of walk(path.join(r, root))) {
    const code = fs.readFileSync(file, "utf8");
    const ast = ts.createSourceFile(file, code, ts.ScriptTarget.Latest, true);
    const kinds = new Set(),
      imports = [];
    function literals(n) {
      if (!n) return [];
      if (ts.isStringLiteral(n)) return [n.text];
      if (ts.isAsExpression(n) || ts.isTypeAssertionExpression(n))
        return literals(n.expression);
      if (ts.isLiteralTypeNode(n)) return literals(n.literal);
      if (ts.isUnionTypeNode(n)) return n.types.flatMap(literals);
      return [];
    }
    function visit(n) {
      if (
        (ts.isPropertyAssignment(n) || ts.isPropertySignature(n)) &&
        n.name.getText(ast).replace(/[\"']/g, "") === "kind"
      )
        for (const value of literals(n.initializer ?? n.type))
          if (known.has(value)) kinds.add(value);
      if (ts.isImportDeclaration(n) && ts.isStringLiteral(n.moduleSpecifier))
        imports.push(n.moduleSpecifier.text);
      if (
        (ts.isVariableDeclaration(n) &&
          n.name.getText(ast) === "checks" &&
          n.initializer &&
          ts.isObjectLiteralExpression(n.initializer)) ||
        (ts.isPropertyAssignment(n) &&
          n.name.getText(ast) === "checks" &&
          ts.isObjectLiteralExpression(n.initializer))
      ) {
        const obj = n.initializer;
        for (const p of obj.properties) {
          if (
            !ts.isPropertyAssignment(p) &&
            !ts.isShorthandPropertyAssignment(p)
          )
            continue;
          const expr = ts.isPropertyAssignment(p)
            ? p.initializer.getText(ast)
            : p.name.getText(ast);
          const line =
            ast.getLineAndCharacterOfPosition(p.getStart(ast)).line + 1;
          checks.push({
            file: path.relative(r, file),
            line,
            name: p.name.getText(ast),
            expression: expr,
            shorthand: ts.isShorthandPropertyAssignment(p),
            candidates: {
              returnedValue: /\b(report|outcome|envelope|value)\??\./.test(
                expr,
              ),
              route:
                /isDeepStrictEqual\((events|effects|journal|hops|operations)|(?:attempts|effects|journal|writes|events|receipts)\.length\s*===|attempts\[[0-9]/.test(
                  expr,
                ),
              protected: /protected|preserv|committed|mainEventId|seed/i.test(
                p.name.getText(ast) + " " + expr,
              ),
            },
          });
        }
      }
      ts.forEachChild(n, visit);
    }
    visit(ast);
    sources.set(file, {
      kinds: [...kinds],
      imports,
      sha256: crypto.createHash("sha256").update(code).digest("hex"),
    });
  }
const domains = {};
for (const kind of known) {
  const declarations = [...sources]
    .filter(([f, s]) => s.kinds.includes(kind))
    .map(([f]) => path.relative(r, f));
  domains[kind] = {
    entry: entries[kind],
    declarations,
    review:
      "AST evidence inventory; classifications are candidates requiring semantic review, not an admission policy",
  };
}
const result = {
  schema: "natlang.semantic-check-inventory/1",
  generatedAt: new Date().toISOString(),
  domainCount: known.size,
  checkCount: checks.length,
  domains,
  sources: Object.fromEntries(
    [...sources].map(([file, value]) => [path.relative(r, file), value]),
  ),
  checks,
};
fs.writeFileSync(args[1], JSON.stringify(result, null, 2) + "\n");
console.log(
  JSON.stringify(
    {
      domainCount: known.size,
      checkCount: checks.length,
      undeclared: Object.entries(domains)
        .filter(([k, v]) => !v.declarations.length)
        .map(([k]) => k),
      routeCandidates: checks
        .filter((x) => x.candidates.route)
        .map((x) => ({
          file: x.file,
          line: x.line,
          name: x.name,
          expression: x.expression,
        })),
    },
    null,
    2,
  ),
);
