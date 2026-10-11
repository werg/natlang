/**
 * External modules: code a program calls but does not own, such as a service, a simulated world, or a store. It runs
 * in the host, outside the program's codebase. The model sees its declaration (types and doc comments, as a
 * TypeScript declaration file would give them) and can call it, but cannot read or edit its implementation.
 */
import ts from 'typescript';
/** Compile a TypeScript module to run in the host, and derive the declaration the model is shown. */
export function externalModule(name, source) {
    const options = { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 };
    const js = ts.transpileModule(source, { compilerOptions: options, fileName: `${name}.ts` }).outputText;
    const module = { exports: {} };
    new Function('exports', 'module', js)(module.exports, module);
    const declared = ts.transpileDeclaration(source, { compilerOptions: options, fileName: `${name}.ts` }).outputText;
    return { exports: module.exports, declaration: declarationNamespace(name, declared) };
}
/**
 * A module's declarations as the namespace the program reaches it by: `declare namespace board { … }`. Exported
 * declarations become members; types the module keeps to itself stay, since the members' signatures use them.
 */
export function declarationNamespace(name, declarations) {
    const file = ts.createSourceFile(`${name}.d.ts`, declarations, ts.ScriptTarget.ES2022, true);
    const members = [];
    for (const statement of file.statements) {
        if (ts.isExportDeclaration(statement))
            continue;
        const exported = ts.canHaveModifiers(statement) &&
            ts.getModifiers(statement)?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword);
        if (!exported && !ts.isTypeAliasDeclaration(statement) && !ts.isInterfaceDeclaration(statement))
            continue;
        const text = statement.getFullText().replace(/^\s*\n/, '').replace(/\b(export\s+)?declare\s+/, '$1');
        members.push(text.split('\n').map(line => line ? `  ${line}` : line).join('\n'));
    }
    return `declare namespace ${name} {\n${members.join('\n')}\n}`;
}
