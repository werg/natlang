import { adaptedCase, requireValue, equal, evalCall, returnCall } from './common.mjs';

const ATTRIBUTES = new Set(['color','size','shape','material']);
function attribute(op, prefix) {
  const key = op.slice(prefix.length); requireValue(ATTRIBUTES.has(key), 'invalid_scene_attribute'); return key;
}
export function executeScene(nodes, scene) {
  requireValue(Array.isArray(nodes) && nodes.length && Array.isArray(scene?.objects), 'invalid_scene_program');
  const outputs = [], expressions = [];
  for (const [index,node] of nodes.entries()) {
    const op = node.function;
    requireValue(typeof op==='string', 'invalid_scene_operator');
    requireValue(Array.isArray(node.inputs) && node.inputs.every(i=>Number.isSafeInteger(i)&&i>=0&&i<index), 'invalid_scene_reference');
    const args = node.inputs.map(i=>outputs[i]), refs = node.inputs.map(i=>`node_${i}`), v = node.value_inputs ?? [];
    const arity = op === 'scene' ? 0 : ['union','intersect','greater_than','less_than'].includes(op) || op.startsWith('equal_') ? 2 : 1;
    requireValue(args.length === arity, 'invalid_scene_arity');
    requireValue(Array.isArray(v) && v.length === (op.startsWith('filter_') || op==='relate' ? 1 : 0), 'invalid_scene_value_arity');
    let result, expression;
    const selection = array => requireValue(Array.isArray(array) && array.every(i=>Number.isSafeInteger(i)&&i>=0&&i<scene.objects.length), 'invalid_scene_selection');
    if (op === 'scene') { result=scene.objects.map((_,i)=>i); expression='scene.objects.map((_,i)=>i)'; }
    else if (op.startsWith('filter_')) {
      const key=attribute(op,'filter_'); selection(args[0]); requireValue(v.length===1 && typeof v[0]==='string', 'invalid_scene_filter');
      result=args[0].filter(i=>scene.objects[i][key]===v[0]);
      expression=`${refs[0]}.filter(i=>scene.objects[i][${JSON.stringify(key)}]===${JSON.stringify(v[0])})`;
    } else if (op==='unique') {
      selection(args[0]); requireValue(args[0].length===1, 'nonunique_scene_object'); result=args[0][0]; expression=`${refs[0]}[0]`;
    } else if (op==='relate') {
      requireValue(Number.isSafeInteger(args[0]) && args[0]>=0 && args[0]<scene.objects.length && ['left','right','front','behind'].includes(v[0]), 'invalid_scene_relation');
      result=scene.relationships?.[v[0]]?.[args[0]]; selection(result);
      expression=`scene.relationships[${JSON.stringify(v[0])}][${refs[0]}]`;
    } else if (op.startsWith('same_')) {
      const key=attribute(op,'same_'); requireValue(Number.isSafeInteger(args[0]) && args[0]>=0 && scene.objects[args[0]], 'invalid_scene_object');
      result=scene.objects.flatMap((o,i)=>i!==args[0]&&o[key]===scene.objects[args[0]][key]?[i]:[]);
      expression=`scene.objects.map((_,i)=>i).filter(i=>i!==${refs[0]}&&scene.objects[i][${JSON.stringify(key)}]===scene.objects[${refs[0]}][${JSON.stringify(key)}])`;
    } else if (op==='union'||op==='intersect') {
      args.forEach(selection); result=op==='union'?[...new Set(args.flat())].sort((a,b)=>a-b):args[0].filter(i=>args[1].includes(i));
      expression=op==='union'?`[...new Set([...${refs[0]},...${refs[1]}])].sort((a,b)=>a-b)`:`${refs[0]}.filter(i=>${refs[1]}.includes(i))`;
    } else if (op==='count'||op==='exist') {
      selection(args[0]); result=op==='count'?args[0].length:args[0].length>0; expression=`${refs[0]}.length${op==='exist'?'>0':''}`;
    } else if (op.startsWith('query_')) {
      const key=attribute(op,'query_'); requireValue(Number.isSafeInteger(args[0]) && args[0]>=0 && scene.objects[args[0]], 'invalid_scene_object'); result=scene.objects[args[0]][key];
      expression=`scene.objects[${refs[0]}][${JSON.stringify(key)}]`;
    } else if (op.startsWith('equal_')) {
      requireValue(['equal_integer','equal_color','equal_size','equal_shape','equal_material'].includes(op), 'unsupported_scene_equality');
      requireValue(args.every(a=>typeof a===(op==='equal_integer'?'number':'string')), 'invalid_scene_equality_type');
      result=args[0]===args[1]; expression=`${refs[0]}===${refs[1]}`;
    } else if (op==='greater_than'||op==='less_than') {
      requireValue(args.every(a=>typeof a==='number'&&Number.isFinite(a)), 'nonnumeric_scene_comparison');
      result=op==='greater_than'?args[0]>args[1]:args[0]<args[1]; expression=`${refs[0]}${op==='greater_than'?'>':'<'}${refs[1]}`;
    } else throw new Error('unsupported_scene_operator');
    requireValue(['number','boolean','string'].includes(typeof result)||Array.isArray(result), 'invalid_scene_result');
    outputs.push(result); expressions.push(expression);
  }
  return {answer:outputs.at(-1), outputs, expressions};
}
export function clevr(original, info) {
  const sem=original.semantics, result=executeScene(sem.nodes,sem.scene);
  requireValue(equal(result.answer,sem.answer), 'scene_source_answer_mismatch');
  requireValue(typeof sem.question==='string'&&sem.question.trim(), 'missing_scene_question');
  const actions=result.expressions.map((expression,index)=>evalCall(index===result.expressions.length-1?`return ${expression};`:`const node_${index} = ${expression};`));
  const record=adaptedCase(original,info,{family:'clevr',suffix:'scene-contract-v2',
    root:{name:'answer_scene_question', args:{question:'string',scene:'Scene'}, returns:typeof sem.answer,
      instructions:'Answer from the structured scene. Object array indexes are identities. For each relation R, relationships[R][i] lists indexes j for which object j stands in relation R to object i: front means j is in front of i, behind means j is behind i, left means j is left of i, and right means j is right of i. In question wording, matte or dull means material rubber; shiny or metallic means material metal; big or large means size large; tiny or small means size small. Match the named attributes exactly, and require a unique object where the question refers to one. Return only the requested value in the declared primitive type, with no explanation or added punctuation.'},
    files:{'types.ts':'export type SceneObject = {color:string, size:string, shape:string, material:string};\nexport type Scene = {objects:SceneObject[], relationships:Record<string, number[][]>};\n'},
    inputs:{question:sem.question,scene:sem.scene}, expected:sem.answer, actions:[...actions,returnCall(sem.answer)],
    checks:['all_node_dependencies_validated','singular_objects_unique','original_functional_program_answer_agreement']});
  record.scene_program=sem.nodes;
  return record;
}
