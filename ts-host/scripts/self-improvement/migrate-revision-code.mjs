import ts from 'typescript';
/** Rewrite source-storage calls by syntax, preserving all surrounding control flow and values. */
export function migrateRevisionCode(source,{populationTypes=true}={}){
 const file=ts.createSourceFile('revision-migration.ts',source,ts.ScriptTarget.ES2022,true),edits=[];
 const visit=node=>{if(populationTypes&&ts.isTypeReferenceNode(node)&&node.typeName.getText(file)==='Member'){edits.push({start:node.typeName.getStart(file),end:node.typeName.getEnd(),text:'PopulationMember'});}
 if(ts.isCallExpression(node)&&ts.isPropertyAccessExpression(node.expression)&&node.expression.expression.getText(file)==='repository'&&node.arguments.length===1){
  const arg=node.arguments[0].getText(file),method=node.expression.name.text;
  if(method==='put'||method==='get'){edits.push({start:node.getStart(file),end:node.getEnd(),text:method==='put'?`(${arg}).digest`:`folder.at(${arg})`});return;}}
  ts.forEachChild(node,visit);};visit(file);
 for(const edit of edits.sort((a,b)=>b.start-a.start))source=source.slice(0,edit.start)+edit.text+source.slice(edit.end);
 return {source,changes:edits.length};
}
