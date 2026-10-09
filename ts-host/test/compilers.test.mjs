import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { compile, normalize } from '../../applications/dist/compilers/index.js';
import { toolchain, toolchainAvailable, toolchainDeclaration } from '../../applications/dist/compilers/toolchain.js';
import compiler from '../../applications/dist/compilers/compiler.nl.js';
import { scriptedModel } from './support/natlang.mjs';

const skip = !toolchainAvailable() ? 'the compilers toolchain needs llvmlite (set NATLANG_LLVM_PYTHON)'
  : spawnSync('gcc', ['--version']).status !== 0 ? 'the compilers back end needs gcc'
  : process.arch !== 'arm64' ? 'the back end targets AArch64' : false;

const SOURCE = `#include <stdio.h>
int square(int x) { return x * x; }
int main(void) { int v = square(6) + 6; printf("%d\\n", v); return 0; }
`;
const HEADER = '@.str.0 = private unnamed_addr constant [4 x i8] c"%d\\0A\\00" ; "%d\\n" in main\ndeclare i32 @printf(ptr, ...)';
const IR = {
  square: {
    lowered: 'define i32 @square(i32 %x) {\nentry:\n  %x.addr = alloca i32\n  store i32 %x, ptr %x.addr\n  %0 = load i32, ptr %x.addr\n  %1 = load i32, ptr %x.addr\n  %mul = mul nsw i32 %0, %1\n  ret i32 %mul\n}',
    ssa: 'define i32 @square(i32 %x) {\nentry:\n  %mul = mul nsw i32 %x, %x\n  ret i32 %mul\n}',
    wrong: 'define i32 @square(i32 %x) {\nentry:\n  %sum = add nsw i32 %x, %x\n  ret i32 %sum\n}',
  },
  main: {
    lowered: 'define i32 @main() {\nentry:\n  %v = alloca i32\n  %call = call i32 @square(i32 6)\n  %add = add nsw i32 %call, 6\n  store i32 %add, ptr %v\n  %0 = load i32, ptr %v\n  %p = call i32 (ptr, ...) @printf(ptr @.str.0, i32 %0)\n  ret i32 0\n}',
    inlined: 'define i32 @main() {\nentry:\n  %v = alloca i32\n  %mul.i = mul nsw i32 6, 6\n  %add = add nsw i32 %mul.i, 6\n  store i32 %add, ptr %v\n  %0 = load i32, ptr %v\n  %p = call i32 (ptr, ...) @printf(ptr @.str.0, i32 %0)\n  ret i32 0\n}',
    ssa: 'define i32 @main() {\nentry:\n  %call = call i32 @square(i32 6)\n  %add = add nsw i32 %call, 6\n  %p = call i32 (ptr, ...) @printf(ptr @.str.0, i32 %add)\n  ret i32 0\n}',
    folded: 'define i32 @main() {\nentry:\n  %p = call i32 (ptr, ...) @printf(ptr @.str.0, i32 42)\n  ret i32 0\n}',
  },
};
const ASM = {
  data: '\t.section\t.rodata\n.L.str.0:\n\t.asciz\t"%d\\n"',
  square: '\t.text\n\t.globl\tsquare\n\t.p2align\t2\n\t.type\tsquare, %function\nsquare:\n\tmul\tw0, w0, w0\n\tret\n\t.size\tsquare, .-square',
  main: '\t.text\n\t.globl\tmain\n\t.p2align\t2\n\t.type\tmain, %function\nmain:\n\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n' +
    '\tmov\tw0, #6\n\tbl\tsquare\n\tadd\tw1, w0, #6\n\tadrp\tx0, .L.str.0\n\tadd\tx0, x0, :lo12:.L.str.0\n\tbl\tprintf\n\tmov\tw0, #0\n\tldp\tx29, x30, [sp], #16\n\tret\n\t.size\tmain, .-main',
};

/** Syntax trees as the front end stages pass them along (types filled by analysis in a real run). */
const node = (kind, line, text, children = []) => ({ kind, line, text, type: null, ref: null, children });
const TREES = [node('FunctionDecl', 2, 'square', [node('ParmVarDecl', 2, 'x'), node('CompoundStmt', 2, null)]),
  node('FunctionDecl', 3, 'main', [node('CompoundStmt', 3, null)])];
/** Machine code between selection and emission: what select, allocate and frame hand on, marked by function. */
const MIR = { square: '; fn square\n\tmul\t%w1, %w0, %w0\n\tmov\tw0, %w1\n\tret', main: '; fn main\n\tbl\tsquare\n\tret' };

/** The interpreter of every stage, scripted: which stage is asked is read from its instructions. */
function compilerModel(pipeline = null, { header = HEADER } = {}) {
  const seen = [], retries = [];
  const model = scriptedModel(opening => {
    if (pipeline && opening.includes('Compile source, a program in language')) return pipeline;
    const fn = opening.includes('define i32 @main') || opening.includes('main:') ? 'main' : 'square';
    const answer = value => `return ${JSON.stringify(value)};`;
    // Stages whose input is a tree or machine code choose their answer from it in eval.
    const byFunction = (test, values) => `return ${test} ? ${JSON.stringify(values.main)} : ${JSON.stringify(values.square)};`;
    const stage = [
      ['Parse source, a C99 program', 'parse'], ['Analyze syntax, a parsed C99', 'analyze'], ['Generate the module-level LLVM 22 IR', 'declare'],
      ['by walking fn.tree, as clang', 'lower'], ['Choose the passes', 'plan'], ["as LLVM's DominatorTree", 'flow'],
      ['Promote memory to registers', 'mem2reg'], ["Simplify fn as LLVM's SCCP", 'simplify'], ['Inline calls in fn', 'inline'],
      ['Remove dead code from fn', 'dce'], ['Write the GNU assembler text', 'data'], ['Select AArch64 instructions for fn', 'select'],
      ['Compute the live intervals', 'liveness'], ['Allocate registers for code by linear scan', 'allocate'], ['Lower the frame of code', 'frame'],
      ['Emit the function name', 'emit'], ['Improve the AArch64 function', 'peephole'],
    ].find(([phrase]) => opening.includes(phrase))?.[1];
    const retry = opening.includes('instead of') || opening.includes('the verifier rejected');
    if (retry) retries.push({ stage, fn, opening });
    seen.push(`${stage}:${['parse', 'analyze', 'declare', 'data'].includes(stage) ? '(module)' : fn}${retry ? ':retry' : ''}`);
    switch (stage) {
      case 'parse': return answer({ declarations: TREES, diagnostics: [] });
      case 'analyze': return 'return { declarations: syntax.declarations, diagnostics: [] };';
      case 'declare': return `return { header: ${JSON.stringify(header)}, diagnostics: [], functions: [
        { name: 'square', signature: 'define i32 @square(i32 %x)', tree: checked.declarations[0] },
        { name: 'main', signature: 'define i32 @main()', tree: checked.declarations[1] }] };`;
      case 'lower': return byFunction('fn.name === "main"', { main: IR.main.lowered, square: IR.square.lowered });
      // main keeps calling square, so square's passes stay observable (an inlined, folded call would hide them).
      case 'plan': return answer(fn === 'main' ? ['mem2reg'] : ['mem2reg', 'simplify', 'dce']);
      case 'flow': return answer({ blocks: [{ name: 'entry', successors: [], predecessors: [], idom: null, frontier: [] }], loops: [] });
      case 'mem2reg': return answer(IR[fn].ssa);
      case 'inline': return answer(IR.main.inlined);
      // The first simplification of square is wrong (it doubles instead of squaring); the retry is right.
      case 'simplify': return answer(fn === 'main' ? IR.main.folded : retry ? IR.square.ssa : IR.square.wrong);
      case 'dce': return answer('define i32 @square(i32 %x) { this is not IR }');
      case 'data': return answer(ASM.data);
      case 'select': return answer(MIR[fn]);
      case 'liveness': return answer({ intervals: [], calls: [] });
      case 'allocate': case 'frame': return byFunction('code.includes("; fn main")', MIR);
      case 'emit': return byFunction('name === "main"', ASM);
      case 'peephole': return byFunction('assembly.includes("main:")', ASM);
    }
    return null;
  });
  return { model, seen, retries };
}

test('module text keeps one declaration per symbol and none for defined functions', () => {
  assert.equal(normalize('declare i32 @f(i32)\ndeclare i32 @printf(ptr, ...)\ndeclare i32 @printf(ptr, ...)\ndefine i32 @f(i32 %a) {\n  ret i32 %a\n}'),
    'declare i32 @printf(ptr, ...)\ndefine i32 @f(i32 %a) {\n  ret i32 %a\n}');
});

// What an interpreter of compiler.nl might write, cut down: every stage is reached through the callable folder.
const PIPELINE = `const syntax = await c.parse(source);
const checked = await c.analyze(syntax);
const frame = await c.declare(checked);
const others = i => frame.functions.filter((_, j) => j !== i).map(f => f.signature.replace(/^define/, 'declare').replace(/ %\\w+/g, ''));
const lowered = await Promise.all(frame.functions.map((fn, i) => c.lower(fn, [frame.header, ...others(i)].join('\\n'))));
const unoptimized = [frame.header, ...lowered].join('\\n');
const checkedIR = await toolchain.verify(unoptimized);
const reference = await toolchain.runIR(unoptimized, inputs[0]);
const ssa = await Promise.all(lowered.map(async (fn, i) => opt.mem2reg(fn, [frame.header, ...others(i)].join('\\n'), await opt.flow(fn))));
const data = await aarch64.data(frame.header);
const emitted = await Promise.all(ssa.map(async (fn, i) => {
  const code = await aarch64.select(fn, [frame.header, ...others(i)].join('\\n'));
  const allocated = await aarch64.allocate(code, await aarch64.liveness(code));
  return aarch64.emit(frame.functions[i].name, await aarch64.frame(allocated));
}));
const assembly = [data, ...emitted].join('\\n');
const ran = await toolchain.runAssembly(assembly, inputs[0]);
return { ir: [frame.header, ...ssa].join('\\n'), assembly, diagnostics: [],
  log: ['verify ' + checkedIR.ok, 'reference ' + JSON.stringify(reference.stdout), 'assembly ' + JSON.stringify(ran.stdout)] };`;

test('the pure pipeline reaches every stage and the toolchain through its callable folder', { skip }, async () => {
  const { model, seen } = compilerModel(PIPELINE);
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const result = await runtime.run(() => compiler(SOURCE, 'c', 'O2', ['']), { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  assert.deepEqual(result.log, ['verify true', 'reference "42\\n"', 'assembly "42\\n"']);
  const stages = seen.map(entry => entry.replace(/:\(module\)$/, ''));
  for (const stage of ['parse', 'analyze', 'declare', 'lower:square', 'mem2reg:square', 'mem2reg:main', 'flow:main', 'data', 'select:square', 'select:main',
    'liveness:square', 'allocate:square', 'frame:square', 'emit:square'])
    assert.ok(stages.includes(stage), `${stage} in ${seen.join(' ')}`);
  assert.equal((await toolchain.runAssembly(result.assembly)).stdout, '42\n');
});

// The pass manager as compiler.nl words it, written the way an interpreter of it would: each answer is checked, a wrong
// one is asked for again with the problem, and a second wrong one is dropped (middle end) or ends the compilation (front end).
const CHECKED_MIDDLE = `const syntax = await c.parse(source);
const checked = await c.analyze(syntax);
const frame = await c.declare(checked);
const others = i => frame.functions.filter((_, j) => j !== i).map(f => f.signature.replace(/^define/, 'declare').replace(/ %\\w+/g, ''));
const lowered = await Promise.all(frame.functions.map((fn, i) => c.lower(fn, [frame.header, ...others(i)].join('\\n'))));
const moduleOf = fns => [frame.header, ...fns].join('\\n');
const reference = await toolchain.runIR(moduleOf(lowered), inputs[0]);
const log = [];
// The check of a middle-end function: the module with the new version verifies and gives the reference output.
const problemWith = async fns => {
  const verdict = await toolchain.verify(moduleOf(fns));
  if (!verdict.ok) return 'the verifier rejected the module: ' + verdict.error;
  const ran = await toolchain.runIR(moduleOf(fns), inputs[0]);
  return ran.stdout === reference.stdout ? null : 'the program printed ' + JSON.stringify(ran.stdout) + ' instead of ' + JSON.stringify(reference.stdout);
};
const pass = async (index, name, ask) => {
  const withText = text => lowered.map((fn, j) => j === index ? text : fn);
  const first = await ask(undefined);
  const problem = await problemWith(withText(first));
  if (!problem) { log.push(name + ' accepted'); return first; }
  const second = await ask(problem);
  const again = await problemWith(withText(second));
  if (!again) { log.push(name + ' accepted after retry: ' + problem); return second; }
  log.push(name + ' kept the previous version: ' + again);
  return lowered[index];
};
const squareContext = [frame.header, ...others(0)].join('\\n');
const ssa = await pass(0, 'mem2reg square', problem => opt.mem2reg(lowered[0], squareContext, { blocks: [], loops: [] }));
lowered[0] = ssa;
const simplified = await pass(0, 'simplify square', problem => problem ? opt.simplify(lowered[0], squareContext, problem) : opt.simplify(lowered[0], squareContext));
lowered[0] = simplified;
const cleaned = await pass(0, 'dce square', problem => problem ? opt.dce(lowered[0], squareContext, problem) : opt.dce(lowered[0], squareContext));
lowered[0] = cleaned;
return { ir: moduleOf(lowered), assembly: '', diagnostics: [], log };`;

test('a stage whose check fails once is asked again with the problem; one that fails twice keeps the previous version', { skip }, async () => {
  const { model, seen, retries } = compilerModel(CHECKED_MIDDLE);
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const run = fn => runtime.run(fn, { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  const result = await compile(SOURCE, { language: 'c', level: 'O2', run, backend: false });
  assert.equal(result.ok, true, result.diagnostics.join('\n'));
  assert.deepEqual(result.log.map(line => line.replace(/:.*/s, '')), ['mem2reg square accepted', 'simplify square accepted after retry', 'dce square kept the previous version']);
  assert.match(result.log[1], /printed "18\\n" instead of "42\\n"/, 'the retry names how the output changed');
  assert.match(result.log[2], /the verifier rejected the module/);
  // The second ask carries the problem; the first does not.
  const simplify = retries.filter(item => item.stage === 'simplify');
  assert.equal(simplify.length, 1);
  assert.match(simplify[0].opening, /problem: string \| undefined = .*printed .*18.* instead of .*42/, 'the retry carries the problem as the problem argument');
  assert.equal(seen.filter(entry => entry === 'simplify:square').length, 1);
  assert.deepEqual(retries.filter(item => item.stage === 'dce').length, 1, 'a stage that failed twice was asked exactly twice');
  assert.equal(seen.filter(entry => entry.startsWith('dce:')).length, 2);
  assert.match(result.ir, /define i32 @square\(i32 %x\) \{\nentry:\n  %mul = mul nsw i32 %x, %x/, 'square keeps the last accepted version');
  assert.equal((await toolchain.runIR(result.ir)).stdout, '42\n');
});

const FRONT_END_PIPELINE = `const syntax = await c.parse(source);
const checked = await c.analyze(syntax);
const asked = [];
let frame = await c.declare(checked);
let verdict = await toolchain.verify(frame.header);
if (!verdict.ok) {
  asked.push(verdict.error);
  frame = await c.declare(checked, 'the verifier rejected the header: ' + verdict.error);
  verdict = await toolchain.verify(frame.header);
  if (!verdict.ok) return { ir: '', assembly: '', diagnostics: ['declare failed twice: ' + verdict.error], log: asked };
}
return { ir: frame.header, assembly: '', diagnostics: [], log: asked };`;

test('a front-end stage that fails twice stops the compilation with its diagnostics', { skip }, async () => {
  const { model, seen, retries } = compilerModel(FRONT_END_PIPELINE, { header: 'this is not IR' });
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const run = fn => runtime.run(fn, { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  const result = await compile(SOURCE, { language: 'c', level: 'O2', run, backend: false });
  assert.equal(result.ok, false);
  assert.match(result.diagnostics[0], /declare failed twice/);
  assert.deepEqual(result.records.map(r => [r.stage, r.accepted]), [['compiler.nl', false]]);
  assert.deepEqual(seen.filter(entry => entry.startsWith('declare')), ['declare:(module)', 'declare:(module):retry']);
  assert.match(retries[0].opening, /the verifier rejected the header/, 'the retry carries the verifier message');
  assert.equal(result.ir, undefined, 'nothing is returned past a failed front end');
});

test('the crisp verifier rejects a final program that misbehaves, whatever the stages accepted', { skip }, async () => {
  // The scripted orchestrator skips every check and emits an assembly whose square doubles instead of squaring.
  const wrongAssembly = ASM.square.replace('mul\tw0, w0, w0', 'add\tw0, w0, w0');
  const pipeline = PIPELINE.replace("const assembly = [data, ...emitted].join('\\n');", `const assembly = [data, ${JSON.stringify(wrongAssembly)}, ${JSON.stringify(ASM.main)}].join('\\n');`);
  assert.notEqual(pipeline, PIPELINE, 'the scripted pipeline was changed');
  const { model } = compilerModel(pipeline);
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const run = fn => runtime.run(fn, { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  const result = await compile(SOURCE, { language: 'c', level: 'O2', run, backend: true });
  assert.equal(result.ok, false);
  assert.equal(result.records.at(-1).stage, 'run assembly');
  assert.equal(result.records.at(-1).accepted, false);
  assert.match(result.diagnostics[0], /the generated program misbehaves: .*printed "18\\n".*instead of "42\\n"/);
});

// Rust: the third front end, sharing the runtime stage with Python. The header declares the vector runtime, so the
// runtime stage writes it; plan picks no passes, and the back end is left out to keep the program small.
const RUST = `fn square(x: i64) -> i64 { x * x }
fn main() { let mut v = Vec::new(); v.push(square(6) + 6); println!("{}", v[0]); }
`;
const RUST_HEADER = ['%vec = type { i64, i64, ptr }',
  '@.str.0 = private unnamed_addr constant [5 x i8] c"%ld\\0A\\00" ; "{}\\n" in main', 'declare i32 @printf(ptr, ...)',
  'declare ptr @rt_vec_new(i64) ; a new empty vector with room for the capacity',
  'declare void @rt_vec_push(ptr, i64) ; append an element', 'declare i64 @rt_vec_get(ptr, i64) ; the element at index'].join('\n');
const RUST_RUNTIME = `declare ptr @malloc(i64)
define ptr @rt_vec_new(i64 %cap) {
entry:
  %v = call ptr @malloc(i64 24)
  %data = call ptr @malloc(i64 64)
  store i64 0, ptr %v
  %capp = getelementptr inbounds %vec, ptr %v, i32 0, i32 1
  store i64 8, ptr %capp
  %datap = getelementptr inbounds %vec, ptr %v, i32 0, i32 2
  store ptr %data, ptr %datap
  ret ptr %v
}
define void @rt_vec_push(ptr %v, i64 %x) {
entry:
  %len = load i64, ptr %v
  %datap = getelementptr inbounds %vec, ptr %v, i32 0, i32 2
  %data = load ptr, ptr %datap
  %slot = getelementptr inbounds i64, ptr %data, i64 %len
  store i64 %x, ptr %slot
  %next = add i64 %len, 1
  store i64 %next, ptr %v
  ret void
}
define i64 @rt_vec_get(ptr %v, i64 %i) {
entry:
  %datap = getelementptr inbounds %vec, ptr %v, i32 0, i32 2
  %data = load ptr, ptr %datap
  %slot = getelementptr inbounds i64, ptr %data, i64 %i
  %x = load i64, ptr %slot
  ret i64 %x
}`;
const RUST_IR = {
  square: 'define i64 @square(i64 %x) {\nentry:\n  %m = mul i64 %x, %x\n  ret i64 %m\n}',
  main: 'define i32 @main() {\nentry:\n  %v = call ptr @rt_vec_new(i64 0)\n  %s = call i64 @square(i64 6)\n  %a = add i64 %s, 6\n' +
    '  call void @rt_vec_push(ptr %v, i64 %a)\n  %x = call i64 @rt_vec_get(ptr %v, i64 0)\n' +
    '  %p = call i32 (ptr, ...) @printf(ptr @.str.0, i64 %x)\n  ret i32 0\n}',
};

test('compile runs compiler.nl and verifies its module and program; the host holds no pass-manager policy', { skip }, async () => {
  const { model } = compilerModel(PIPELINE);
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const run = fn => runtime.run(fn, { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  const result = await compile(SOURCE, { language: 'c', level: 'O2', run, backend: true });
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.ok, true);
  assert.deepEqual(result.records.map(r => [r.stage, r.accepted]), [['compiler.nl', true], ['verify', true], ['run ir', true], ['run assembly', true]]);
  assert.deepEqual(result.log, ['verify true', 'reference "42\\n"', 'assembly "42\\n"']);
  // An expected output the program does not print is a failure of the final check, not of the pass manager.
  const wrong = await compile(SOURCE, { language: 'c', level: 'O2', run, backend: true, expected: ['41\n'] });
  assert.equal(wrong.ok, false);
  assert.match(wrong.diagnostics[0], /printed "42\\n".*instead of "41\\n"/);
  assert.equal(wrong.records.at(-1).stage, 'run ir');
});

const RUST_PIPELINE = `const syntax = await rust.parse(source);
const checked = await rust.analyze(syntax);
const frame = await rust.declare(checked);
const runtimeText = await runtime(frame.header);
const header = frame.header + '\\n' + runtimeText;
const lowered = await Promise.all(frame.functions.map(fn => rust.lower(fn, header)));
return { ir: [header, ...lowered].join('\\n'), assembly: '', diagnostics: [], log: [] };`;

test('a Rust program goes through the Rust front end and the shared runtime stage', { skip }, async () => {
  const seen = [];
  const model = scriptedModel(opening => {
    const answer = value => `return ${JSON.stringify(value)};`;
    if (opening.includes('Compile source, a program in language')) return RUST_PIPELINE;
    if (opening.includes('Parse source, a Rust 2021 program')) { seen.push('parse');
      return answer({ declarations: [node('ItemFn', 1, 'square'), node('ItemFn', 2, 'main')], diagnostics: [] }); }
    if (opening.includes('Analyze syntax, a parsed Rust program')) { seen.push('analyze'); return 'return { declarations: syntax.declarations, diagnostics: [] };'; }
    if (opening.includes('an analyzed Rust program')) { seen.push('declare'); return `return { header: ${JSON.stringify(RUST_HEADER)}, diagnostics: [], functions: [
      { name: 'square', signature: 'define i64 @square(i64 %x)', tree: checked.declarations[0] },
      { name: 'main', signature: 'define i32 @main()', tree: checked.declarations[1] }] };`; }
    if (opening.includes('header declares the runtime functions')) { seen.push('runtime'); return answer(RUST_RUNTIME); }
    if (opening.includes('a shadowing `let`')) { seen.push('lower');
      return `return fn.name === 'main' ? ${JSON.stringify(RUST_IR.main)} : ${JSON.stringify(RUST_IR.square)};`; }
    if (opening.includes('Choose the passes')) return answer([]);
    return null;
  });
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const run = fn => runtime.run(fn, { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  const result = await compile(RUST, { language: 'rust', level: 'O1', run, backend: false });
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.ok, true);
  assert.deepEqual(seen.sort(), ['analyze', 'declare', 'lower', 'lower', 'parse', 'runtime']);
  assert.match(result.ir, /define ptr @rt_vec_new\(i64 %cap\)/, 'the runtime is part of the module');
  assert.equal((await toolchain.runIR(result.ir)).stdout, '42\n');
});
