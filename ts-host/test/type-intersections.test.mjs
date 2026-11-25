import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType } from '../dist/index.js';
import { typeScriptText } from '../dist/compiler/eval-check.js';
import { coerce as coerceValue } from '../dist/native/values.js';
const coerce = (value, type, env = new TypeEnv()) => coerceValue(value, type, env);
import { finiteValues, decisionDomain, domainValues } from '../dist/native/decision.js';
import { expandType } from '../dist/native/expand-type.js';

test('intersections of object types parse to one merged record', () => {
  assert.equal(formatType(parseType('{ a: number } & { b?: string }')), '{ a: number, b?: string }');
  assert.equal(formatType(parseType('{ a: number } & { a: number, c: boolean }')), '{ a: number, c: boolean }');
});

test('an alias in an intersection merges when it is resolved, and fits like the merged record', () => {
  const env = new TypeEnv({ Base: parseType('{ id: string }'), Dict: parseType('{ [k: string]: number }') });
  const type = parseType('Base & { note: string }');
  assert.equal(formatType(type), 'Base & { note: string }');
  assert.equal(formatType(env.resolve(type)), '{ id: string, note: string }');
  assert.equal(fitsType(parseType('{ id: string, note: string }'), type, env), true);
  assert.equal(fitsType(type, parseType('Base'), env), false, 'a record with extra fields does not fit the narrower record');
  assert.equal(fitsType(type, parseType('{ id: string, note: string }'), env), true);
  assert.equal(formatType(env.resolve(parseType('Dict & { a: number }'))), '{ a: number }');
});

test('string index signatures are dictionaries', () => {
  assert.equal(formatType(parseType('{ [k: string]: number }')), 'Record<string, number>');
  assert.deepEqual(parseType('{ [key: string]: { n: number }[] }').kind, 'dict');
  assert.throws(() => parseType('{ [k: number]: string }'), /string keys/);
  assert.throws(() => parseType('{ a: number, [k: string]: number }'), /bad|expected/);
  assert.throws(() => parseType('{ [k: string]: number, a: number }'), /stands alone/);
  assert.equal(fitsType(parseType('{ [k: string]: "a" }'), parseType('Record<string, string>')), true);
});

test('indexed access types resolve to the field type', () => {
  const env = new TypeEnv({ State: parseType('{ status: "open" | "closed", note?: string, tags: Record<string, number> }') });
  assert.equal(formatType(env.resolve(parseType("State['status']"))), '"open" | "closed"');
  assert.equal(formatType(env.resolve(parseType('State["note"]'))), 'string | null');
  assert.equal(formatType(env.resolve(parseType("State['tags']['x']"))), 'number');
  assert.equal(formatType(parseType("{ a: number }['a']")), 'number');
  assert.equal(fitsType(parseType('"open"'), parseType("State['status']"), env), true);
  assert.equal(fitsType(parseType('"stale"'), parseType("State['status']"), env), false);
  assert.throws(() => env.resolve(parseType("State['missing']")), /no field "missing"/);
  assert.equal(formatType(parseType("State['status']")), 'State["status"]');
});

test('indexed access resolves a common field of a union', () => {
  const env = new TypeEnv({
    ReaderSelection: parseType('{ kind: "inbox" | "saved" | "backlog" } | { kind: "archive", query: string }'),
    Part: parseType('{ value?: number } | Record<string, boolean>'),
    Nested: parseType('Part | { value: string }'),
    Recursive: parseType('Recursive | { value: string }'),
  });
  const kind = parseType('ReaderSelection["kind"]');
  assert.equal(formatType(env.resolve(kind)), '"inbox" | "saved" | "backlog" | "archive"');
  assert.equal(fitsType(parseType('"archive"'), kind, env), true);
  assert.equal(fitsType(parseType('"missing"'), kind, env), false);
  assert.throws(() => env.resolve(parseType('ReaderSelection["query"]')), /no field "query"/);
  assert.equal(formatType(env.resolve(parseType('Nested["value"]'))), 'number | null | boolean | string');
  assert.equal(formatType(env.resolve(parseType('({ item: { name: string } } | { item: Record<string, number> })["item"]["name"]'))), 'string | number');
  assert.throws(() => env.resolve(parseType('Recursive["value"]')), /recursive union/);
});

test('the same types are valid TypeScript in declarations', () => {
  for (const text of ['A & { b: string }', '{ [k: string]: number }', "State['status']"])
    assert.equal(typeScriptText(text, new Set()), text);
});

test('Record keys: string-literal unions are exact records, string and aliased keys are dictionaries', () => {
  assert.equal(formatType(parseType('Record<"blocking" | "praise", number>')), '{ blocking: number, praise: number }');
  assert.equal(formatType(parseType("Record<'only', string>")), '{ only: string }');
  assert.equal(formatType(parseType('Record<Kind, number>')), 'Record<Kind, number>');
  assert.equal(formatType(parseType('Record<string, number>')), 'Record<string, number>');
  assert.throws(() => parseType('Record<number, string>'), /Record keys are strings/);
  assert.throws(() => parseType('Record<1 | 2, string>'), /Record keys are strings/);
});


test('literal intersections narrow unions and primitive slots without widening impossible alternatives', () => {
  const env = new TypeEnv({ Status: parseType('"matched" | "absent" | "mismatch" | "incomplete"') });
  for (const text of ['Status & "matched"', '"matched" & Status', 'string & "matched"',
    '("matched" | "absent") & ("matched" | "mismatch")', 'unknown & "matched"']) {
    assert.equal(formatType(env.resolve(parseType(text))), '"matched"', text);
    assert.equal(coerce('matched', parseType(text), env), 'matched');
    assert.throws(() => coerce('absent', parseType(text), env));
  }
  for (const text of ['string & number', '"matched" & "absent"', 'Status & "other"', 'number & "matched"', 'null & { id:string }']) {
    const type = parseType(text);
    assert.equal(formatType(env.resolve(type)), 'never', text);
    for (const value of [null, 'matched', 'absent', 1, {}]) assert.throws(() => coerce(value, type, env));
    assert.equal(fitsType(type, parseType('string'), env), true);
  }
  assert.throws(() => env.resolve(parseType('Blob & string')), /unsupported structural intersection/);
  assert.throws(() => env.resolve(parseType('Live<"Opaque", "any"> & string')), /unsupported structural intersection/);
  assert.equal(formatType(parseType('never')), 'never');
  assert.equal(fitsType(parseType('string'), parseType('never')), false);
});

test('record narrowing preserves literal restrictions, non-null observations, requiredness and exact fields', () => {
  const env = new TypeEnv({ Review: parseType('{ journal: "matched" | "absent"; effect: "confirmed" | "unchecked"; readback: { digest:string; observedDigest:string|null; size:number|null } | null }') });
  const type = parseType('Review & { journal:"matched"; effect:"confirmed"; readback:{digest:string;observedDigest:string;size:number} }');
  const value = { journal:'matched', effect:'confirmed', readback:{digest:'receipt',observedDigest:'receipt',size:115} };
  assert.deepEqual(coerce(value, type, env), value);
  for (const change of [r=>r.journal='absent',r=>r.effect='unchecked',r=>r.readback=null,
    r=>r.readback.observedDigest=null,r=>r.readback.size=null,r=>r.extra=1]) {
    const bad=structuredClone(value);change(bad);assert.throws(()=>coerce(bad,type,env));
  }
  const missing=structuredClone(value);delete missing.readback;
  assert.throws(()=>coerceValue(missing,type,env,'value',{preserveRecordExtras:true}),'callable input mode requires every required field');
  const expanded=expandType(type,env);
  assert.equal(expanded.kind,'record');
  assert.equal(formatType(expanded.fields.find(f=>f.name==='journal').type),'"matched"');
  assert.equal(formatType(env.resolve(parseType('{ value?: string } & { value: "matched" }'))),'{ value: "matched" }');
  const optional = parseType('{ value?: string } & { value?: number }');
  assert.deepEqual(coerce({},optional),{});
  assert.throws(()=>coerce({value:1},optional));
  assert.equal(formatType(env.resolve(parseType('{ value: string } & { value: number }'))),'never');
});

test('union distribution removes impossible discriminated records and retains the exact supported variant', () => {
  const env=new TypeEnv();
  const type=parseType('({kind:"matched";value:number} | {kind:"absent";reason:string}) & {kind:"matched"}');
  assert.equal(formatType(env.resolve(type)),'{ kind: "matched", value: number }');
  assert.deepEqual(coerce({kind:'matched',value:2},type),{kind:'matched',value:2});
  assert.throws(()=>coerce({kind:'absent',reason:'not read'},type));
  const recursive=new TypeEnv({ Recursive:parseType('Recursive | "matched"') });
  assert.throws(()=>recursive.resolve(parseType('Recursive & "matched"')),/recursive union/);
});

test('dictionary, list and tuple constraints remain conjunctive, including empty collections and optional positions', () => {
  const env=new TypeEnv();
  const dict=parseType('Record<string,string> & { code:"matched" }');
  assert.deepEqual(coerce({code:'matched'},dict),{code:'matched'});
  assert.throws(()=>coerce({code:'absent'},dict));
  assert.throws(()=>coerce({code:'matched',extra:'value'},dict));
  assert.deepEqual(coerce([],parseType('string[] & number[]')),[]);
  assert.throws(()=>coerce(['value'],parseType('string[] & number[]')));
  assert.equal(formatType(env.resolve(parseType('[string] & [number]'))),'never');
  assert.deepEqual(coerce([],parseType('[string?] & [number?]')),[]);
  assert.throws(()=>coerce([1],parseType('[string?] & [number?]')));
  assert.deepEqual(coerce(['matched'],parseType('string[] & ["matched", number?]')),['matched']);
});

test('intersection normalization keeps refinement predicates and untrusted wrappers', () => {
  const env=new TypeEnv();
  assert.equal(formatType(env.resolve(parseType('Is<string,"checked"> & "matched"'))),'Is<"matched", "checked">');
  assert.equal(formatType(env.resolve(parseType('Is<string,"checked"> & Is<string,"current">'))),'Is<Is<string, "checked">, "current">');
  assert.equal(formatType(env.resolve(parseType('Untrusted<string> & "matched"'))),'Untrusted<"matched">');
});

test('finite decision domains resolve intersections, indexed slots and utilities through the shared normalizer', () => {
  const env=new TypeEnv({ Status:parseType('"matched" | "absent"'), Result:parseType('{status:Status;ok:boolean}') });
  assert.deepEqual(finiteValues(parseType('Status & "matched"'),env),['matched']);
  assert.deepEqual(finiteValues(parseType('Result["status"]'),env),['matched','absent']);
  assert.deepEqual([...domainValues(decisionDomain(parseType('Result & {status:"matched"}'),env))].map(x=>x.value),[
    {status:'matched',ok:true},{status:'matched',ok:false}]);
  assert.deepEqual([...domainValues(decisionDomain(parseType('Pick<Result,"status">'),env))].map(x=>x.value),[
    {status:'matched'},{status:'absent'}]);
  assert.equal(decisionDomain(parseType('"matched" & "absent"'),env).size,0);
});


test('aliased discriminants discard impossible variants before callback signature fitting', () => {
  const env=new TypeEnv({ Good:parseType('"matched"'), Bad:parseType('"absent"'),
    Result:parseType('{kind:Good;value:number} | {kind:Bad;reason:string}'),
    RecursiveA:parseType('{value:number;next?:RecursiveA}'),RecursiveB:parseType('{value:number;next?:RecursiveB}') });
  const narrowed=parseType('Result & {kind:"matched"}');
  assert.equal(formatType(env.resolve(narrowed)),'{ kind: "matched", value: number }');
  assert.equal(fitsType(narrowed,parseType('{kind:"matched";value:number}'),env),true);
  assert.deepEqual(coerce({value:1},parseType('RecursiveA & RecursiveB'),env),{value:1});
  assert.deepEqual(coerce({value:1,next:{value:2}},parseType('RecursiveA & RecursiveB'),env),{value:1,next:{value:2}});
});
