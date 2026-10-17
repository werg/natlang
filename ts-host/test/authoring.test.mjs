import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkAuthoring } from '../dist/teacher/authoring.js';

const files = {
  'app.ts': 'import type { Ticket } from "./types";\nexport async function urgent(tickets: Ticket[]): Promise<string[]> {\n' +
    '  const flags = await Promise.all(tickets.map(ticket => nl`Does ticket report an outage?`(ticket)));\n' +
    '  return tickets.filter((t, i) => flags[i]).map(t => t.id);\n}\n',
  'types.ts': 'export type Ticket = { id: string, text: string };\n',
};
const spec = { module: 'app.ts', export: 'urgent', requires: { nl: true },
  runs: [{ args: [[{ id: 'a', text: 'site down' }, { id: 'b', text: 'typo' }]], expected: ['a'] }],
  oracle: [{ match: ['"a"'], value: true }, { match: ['"b"'], value: false }] };

test('authored code is judged by running it with an oracle for its nl children', async () => {
  // `import type ... from "./types"` resolves to the folder's aliases, so the child is typed (ticket: Ticket).
  assert.deepEqual(await checkAuthoring(files, spec), { ok: true, problems: [], results: [['a']] });
  const wrong = await checkAuthoring({ ...files, 'app.ts': files['app.ts'].replace('flags[i]', '!flags[i]') }, spec);
  assert.deepEqual(wrong.problems, ['run 0: got ["b"], expected ["a"]']);
  const regex = await checkAuthoring({ ...files, 'app.ts': 'export async function urgent(tickets: { id: string, text: string }[]): Promise<string[]> {\n  return tickets.filter(t => /down/.test(t.text)).map(t => t.id);\n}\n' }, spec);
  assert.deepEqual(regex.problems, ['app.ts creates no nl function']);
});

test('source-derived oracle aliases cover full ticket text without id and preserve original tests',async()=>{
  const {withSourceArgumentAliases}=await import('../dist/teacher/authoring.js');
  const textOnly={...files,'app.ts':files['app.ts'].replace('`(ticket)','`(ticket.text)')};
  const legacy=await checkAuthoring(textOnly,spec);assert.equal(legacy.ok,false);
  const grounded=withSourceArgumentAliases(spec);
  assert.equal('argument_aliases'in spec,false);
  assert.deepEqual(grounded.runs,spec.runs);assert.deepEqual(grounded.oracle,spec.oracle);
  assert.deepEqual(await checkAuthoring(textOnly,grounded),{ok:true,problems:[],results:[['a']]});
  const forged=structuredClone(grounded);forged.argument_aliases[0].value=false;
  assert.equal((await checkAuthoring(textOnly,forged)).ok,false);
  assert.equal((await checkAuthoring({...textOnly,'app.ts':textOnly['app.ts'].replace('flags[i]','!flags[i]')},grounded)).ok,false);
});

test('source alias derivation rejects conflicting fulltext and existing matching oracles',async()=>{
  const {withSourceArgumentAliases}=await import('../dist/teacher/authoring.js');
  const duplicate=structuredClone(spec);duplicate.runs[0].args[0][1].text='site down';
  assert.throws(()=>withSourceArgumentAliases(duplicate),/Conflicting/);
  const oldMatch=structuredClone(spec);oldMatch.oracle.push({match:[JSON.stringify('site down')],value:false});
  assert.throws(()=>withSourceArgumentAliases(oldMatch),/Conflicting/);
  const partial=structuredClone(spec);partial.runs[0].args[0][0].extra='context';
  assert.equal(withSourceArgumentAliases(partial).argument_aliases.some(item=>item.source_id==='a'),false);
});

test('fresh authoring constructor declares grounded aliases under a new task identity',async()=>{
  const {authoringInlineReview}=await import('../scripts/inline-curriculum/authoring.mjs');
  const [program]=authoringInlineReview(180017,2);
  assert.match(program.id,/:source-argument-oracle-v1$/);
  assert.equal(program.generation.authoring_oracle,'source-run-arguments/1');
  assert.equal(program.semantics.authoring.argument_aliases.length>0,true);
  const source=program.curriculum.reference.root.find(call=>call[0]==='write_file')[1];
  const actual={...program.semantics.folder_files,[source.path]:source.content};
  assert.equal((await checkAuthoring(actual,program.semantics.authoring)).ok,true);
  actual[source.path]=source.content.replace('`(ticket)','`(ticket.text)');
  assert.equal((await checkAuthoring(actual,program.semantics.authoring)).ok,true);
});


test('forged aliases cannot authenticate themselves when no source aliases are derivable',async()=>{
  const {withSourceArgumentAliases}=await import('../dist/teacher/authoring.js');
  const forged={...structuredClone(spec),runs:[{args:[['no source Ticket']],expected:[]}],
    argument_aliases:[{match:['invented'],value:true,source_id:'invented',source_text:'invented'}]};
  assert.equal('argument_aliases'in withSourceArgumentAliases(forged),false);
  assert.deepEqual(await checkAuthoring(files,forged),{ok:false,problems:['Source argument oracle aliases differ from the original declared runs.'],results:[]});
});

test('nl.with satisfies the real construct requirement and legacy false-negative stays reproducible',async()=>{
  const {checkAuthoringLegacyRequirements}=await import('../dist/teacher/authoring.js');
  const typed={...files,'app.ts':files['app.ts'].replace('  const flags',
    '  const classify: Neuralese<(ticket: Ticket) => Promise<boolean>> = nl.with<boolean>({})`Does ticket report an outage?`;\n  const flags')
    .replace('nl`Does ticket report an outage?`(ticket)','classify(ticket)')};
  assert.deepEqual(await checkAuthoring(typed,spec),{ok:true,problems:[],results:[['a']]});
  assert.deepEqual(await checkAuthoringLegacyRequirements(typed,spec),{ok:false,problems:['app.ts creates no nl function'],results:[['a']]});
  assert.deepEqual((await checkAuthoring(typed,{...spec,requires:{noNl:true}})).problems,
    ['app.ts creates an nl function where an existing one fits']);
});

test('comments and string contents cannot impersonate an authored nl function',async()=>{
  for(const suffix of ['// nl`pretend child`','const unused = "nl`pretend child`";']){
    const crisp={'app.ts':'export async function urgent(tickets: {id:string,text:string}[]): Promise<string[]> { return tickets.filter(t=>t.id==="a").map(t=>t.id); }\n'+suffix};
    assert.deepEqual((await checkAuthoring(crisp,spec)).problems,['app.ts creates no nl function']);
    assert.equal((await checkAuthoring(crisp,{...spec,requires:{noNl:true}})).ok,true);
  }
});
