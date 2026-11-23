import ts from 'typescript';

/** Link explicit type imports/re-exports using the compiler's parser, leaving
 * runtime imports, comments and source text that merely mentions an import alone.
 */
export function rewriteTypeModuleSpecifiers(source:string, resolve:(specifier:string)=>string|undefined, fileName="types.ts"):string {
  const file=ts.createSourceFile(fileName,source,ts.ScriptTarget.Latest,true,fileName.endsWith('.tsx')?ts.ScriptKind.TSX:ts.ScriptKind.TS);
  const edits=new Map<number,{start:number;end:number;text:string}>();
  const link=(literal:ts.StringLiteralLike)=>{
    const value=resolve(literal.text);
    if(value!==undefined&&value!==literal.text)edits.set(literal.getStart(file),{start:literal.getStart(file),end:literal.end,text:JSON.stringify(value)});
  };
  const visit=(node:ts.Node):void=>{
    if(ts.isImportDeclaration(node)&&ts.isStringLiteral(node.moduleSpecifier)){
      const clause=node.importClause,binding=clause?.namedBindings;
      if(clause?.isTypeOnly||clause&&!clause.name&&binding&&ts.isNamedImports(binding)&&binding.elements.length>0&&binding.elements.every(item=>item.isTypeOnly))link(node.moduleSpecifier);
    }else if(ts.isExportDeclaration(node)&&node.moduleSpecifier&&ts.isStringLiteral(node.moduleSpecifier)){
      if(node.isTypeOnly||node.exportClause&&ts.isNamedExports(node.exportClause)&&node.exportClause.elements.length>0&&node.exportClause.elements.every(item=>item.isTypeOnly))link(node.moduleSpecifier);
    }else if(ts.isImportTypeNode(node)&&ts.isLiteralTypeNode(node.argument)&&ts.isStringLiteral(node.argument.literal))link(node.argument.literal);
    else if(ts.isImportEqualsDeclaration(node)&&node.isTypeOnly&&ts.isExternalModuleReference(node.moduleReference)&&node.moduleReference.expression&&ts.isStringLiteral(node.moduleReference.expression))link(node.moduleReference.expression);
    ts.forEachChild(node,visit);
  };
  visit(file);
  let result=source;for(const edit of [...edits.values()].sort((a,b)=>b.start-a.start))result=result.slice(0,edit.start)+edit.text+result.slice(edit.end);
  return result;
}
