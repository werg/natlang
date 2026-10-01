import ts from 'typescript';
/** Add an explicit finite workflow allowance where an old caller provided no terminating primitive. */
export function migrateIterationCode(source,maxSteps=128){
 const file=ts.createSourceFile('migration.ts',source,ts.ScriptTarget.ES2022,true),edits=[];
 const visit=node=>{
  if(ts.isCallExpression(node)&&ts.isPropertyAccessExpression(node.expression)&&node.expression.name.text==='iterateOn'||ts.isCallExpression(node)&&ts.isIdentifier(node.expression)&&node.expression.text==='iterateOn'){
   let current=node,bounded=false;
   while(ts.isPropertyAccessExpression(current.parent)&&ts.isCallExpression(current.parent.parent)){
    const call=current.parent.parent,name=current.parent.name.text;
    if(name==='withMeasure'||name==='withLimit'&&call.arguments.some(arg=>/\bmaxSteps\s*:/.test(arg.getText(file))))bounded=true;
    current=call;
   }
   if(!bounded)edits.push({start:node.getStart(file),end:node.getEnd(),text:node.getText(file)+`.withLimit({maxSteps:${maxSteps}})`});
  }
  ts.forEachChild(node,visit);
 };
 visit(file);for(const edit of edits.sort((a,b)=>b.start-a.start))source=source.slice(0,edit.start)+edit.text+source.slice(edit.end);
 return {source,changed:edits.length};
}
