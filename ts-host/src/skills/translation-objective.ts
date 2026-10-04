/** Exact behavior preservation for newly specified bounded expression languages. */
export type Expression = {variable:'x'|'y'} | {constant:number} | {op:string,args:Expression[]};
export type Language = {operators:Record<string,'add'|'subtract'|'multiply'|'minimum'|'maximum'|'negate'|'absolute'>};
export type TranslationTask = {schema:'natlang.skill-translation/1',id:string,revision:string,language:Language,tests:{input:{x:number,y:number},expected:number}[]};
const ops=new Set(['add','subtract','multiply','minimum','maximum','negate','absolute']);
export function evaluateExpression(program:unknown,input:{x:number,y:number},language:Language):number {
 if(!language||!language.operators||Object.keys(language.operators).length>16||Object.entries(language.operators).some(([name,op])=>!name||!ops.has(op)))throw Error('invalid language');
 if(!Number.isSafeInteger(input.x)||!Number.isSafeInteger(input.y))throw Error('invalid input');
 let count=0;
 const visit=(node:any,depth:number):number=>{
  if(++count>128||depth>16||!node||typeof node!=='object'||Array.isArray(node))throw Error('invalid expression allocation or node');
  let value:number;
  if(Object.keys(node).length===1&&['x','y'].includes(node.variable))value=input[node.variable as 'x'|'y'];
  else if(Object.keys(node).length===1&&Number.isSafeInteger(node.constant)&&Math.abs(node.constant)<=1000)value=node.constant;
  else if(Object.keys(node).length===2&&typeof node.op==='string'&&Object.hasOwn(language.operators,node.op)&&Array.isArray(node.args)){
   const op=language.operators[node.op],arity=['negate','absolute'].includes(op!)?1:2;
   if(node.args.length!==arity)throw Error('operator arity mismatch');
   const values=node.args.map((child:unknown)=>visit(child,depth+1)),a=values[0],b=values[1];
   value=op==='add'?a+b:op==='subtract'?a-b:op==='multiply'?a*b:op==='minimum'?Math.min(a,b):op==='maximum'?Math.max(a,b):op==='negate'?-a:Math.abs(a);
  }else throw Error('invalid expression shape or operator');
  if(!Number.isSafeInteger(value))throw Error('unsafe computational output');return value;
 };
 return visit(program,0);
}
export function scoreSpecifiedTranslation(task:TranslationTask,value:unknown):{quality:number,gates:Record<string,boolean>}{
 if(!task||task.schema!=='natlang.skill-translation/1'||!task.id||!task.revision||!Array.isArray(task.tests)||!task.tests.length||task.tests.length>64||task.tests.some(t=>!Number.isSafeInteger(t.expected)))throw Error('invalid translation reference');
 // Invalid host references are infrastructure failures, never candidate rejections.
 for(const test of task.tests)evaluateExpression({variable:'x'},test.input,task.language);
 let program:unknown;
 try{program=typeof value==='string'?JSON.parse(value):value;}catch{return {quality:0,gates:{parsed:false}};}
 try{
  const right=task.tests.filter(test=>evaluateExpression(program,test.input,task.language)===test.expected).length;
  return {quality:right/task.tests.length,gates:{executable:true,all_correct:right===task.tests.length}};
 }catch{return {quality:0,gates:{executable:false,all_correct:false}};}
}
