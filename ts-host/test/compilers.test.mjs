import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { compile, normalize, splitAssembly } from '../../applications/dist/compilers/index.js';
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

/** The interpreter of every stage, scripted: which stage is asked is read from its instructions. */
function compilerModel(pipeline = null) {
  const seen = [];
  const model = scriptedModel(opening => {
    if (pipeline && opening.includes('Compile source, a program in language')) return pipeline;
    const fn = opening.includes('define i32 @main') || opening.includes('main:') ? 'main' : 'square';
    const answer = value => `return ${JSON.stringify(value)};`;
    const stage = [
      ['semantic analysis does', 'declare'], ['Translate the C function fn', 'lower'], ['Choose the passes', 'plan'],
      ['Promote memory to registers', 'mem2reg'], ["Simplify fn as LLVM's InstCombine", 'simplify'], ['Inline calls in fn', 'inline'],
      ['Remove dead code from fn', 'dce'], ['Write the GNU assembler text', 'data'], ['Translate the LLVM IR function fn into GNU', 'select'],
      ['Allocate registers', 'allocate'], ['Improve the AArch64 function', 'peephole'],
    ].find(([phrase]) => opening.includes(phrase))?.[1];
    const retry = opening.includes('instead of') || opening.includes('the verifier rejected');
    seen.push(`${stage}:${fn}${retry ? ':retry' : ''}`);
    switch (stage) {
      case 'declare': return answer({ header: HEADER, diagnostics: [], functions: [
        { name: 'square', signature: 'define i32 @square(i32 %x)', source: 'int square(int x) { return x * x; }' },
        { name: 'main', signature: 'define i32 @main()', source: 'int main(void) { int v = square(6) + 6; printf("%d\\n", v); return 0; }' }] });
      case 'lower': return answer(IR[fn].lowered);
      // main keeps calling square, so square's passes stay observable (an inlined, folded call would hide them).
      case 'plan': return answer(fn === 'main' ? ['mem2reg'] : ['mem2reg', 'simplify', 'dce']);
      case 'mem2reg': return answer(IR[fn].ssa);
      case 'inline': return answer(IR.main.inlined);
      // The first simplification of square is wrong (it doubles instead of squaring); the retry is right.
      case 'simplify': return answer(fn === 'main' ? IR.main.folded : retry ? IR.square.ssa : IR.square.wrong);
      case 'dce': return answer('define i32 @square(i32 %x) { this is not IR }');
      case 'data': return answer(ASM.data);
      case 'select': return answer(ASM[fn]);
      case 'allocate': case 'peephole': return answer(fn === 'main' ? ASM.main : ASM.square);
    }
    return null;
  });
  return { model, seen };
}

test('module text keeps one declaration per symbol and none for defined functions', () => {
  assert.equal(normalize('declare i32 @f(i32)\ndeclare i32 @printf(ptr, ...)\ndeclare i32 @printf(ptr, ...)\ndefine i32 @f(i32 %a) {\n  ret i32 %a\n}'),
    'declare i32 @printf(ptr, ...)\ndefine i32 @f(i32 %a) {\n  ret i32 %a\n}');
  const { functions, rest } = splitAssembly('\t.text\n\t.globl\tf\n\t.p2align\t2\n\t.type\tf,@function\nf:\n\t.cfi_startproc\n\tret\n.Lfunc_end0:\n\t.size\tf, .Lfunc_end0-f\n\t.cfi_endproc\n\n\t.data\ng:\n\t.xword\t5');
  assert.deepEqual([...functions.keys()], ['f']);
  assert.match(functions.get('f'), /^\t\.globl\tf\n[^]*\.cfi_endproc$/);
  assert.match(rest, /\.data\ng:/);
});

test('every compiler stage is checked by running the program; a wrong stage is retried once, then rejected', { skip }, async () => {
  const { model, seen } = compilerModel();
  const runtime = createNatlangRuntime({ model: model.driver });
  const run = fn => runtime.run(fn, { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  const result = await compile(SOURCE, { language: 'c', level: 'O2', run, backend: true });
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.ok, true);
  const record = (fn, stage) => result.records.find(r => r.function === fn && r.stage === stage);
  assert.deepEqual([record('square', 'simplify').accepted, record('square', 'simplify').attempts], [true, 2], 'retried with the behavior difference');
  assert.deepEqual([record('square', 'dce').accepted, record('square', 'dce').attempts], [false, 2], 'invalid IR is rejected twice');
  assert.ok(seen.includes('simplify:square:retry'));
  assert.match(result.ir, /define i32 @square\(i32 %x\) \{\nentry:\n  %mul = mul nsw i32 %x, %x/, 'square keeps its last accepted version');
  assert.match(result.ir, /define i32 @main\(\) \{\nentry:\n  %call = call i32 @square\(i32 6\)/, 'main is in SSA form');
  for (const stage of ['select', 'allocate', 'peephole']) assert.equal(record('main', stage).accepted, true, stage);
  assert.equal((await toolchain.runAssembly(result.assembly)).stdout, '42\n');
});

// What an interpreter of compiler.nl might write, cut down: every stage is reached through the callable folder.
const PIPELINE = `const frame = await c.declare(source);
const others = i => frame.functions.filter((_, j) => j !== i).map(f => f.signature.replace(/^define/, 'declare').replace(/ %\\w+/g, ''));
const lowered = await Promise.all(frame.functions.map((fn, i) => c.lower(fn, [frame.header, ...others(i)].join('\\n'))));
const unoptimized = [frame.header, ...lowered].join('\\n');
const checked = await toolchain.verify(unoptimized);
const reference = await toolchain.runIR(unoptimized, inputs[0]);
const ssa = await Promise.all(lowered.map((fn, i) => opt.mem2reg(fn, [frame.header, ...others(i)].join('\\n'))));
const data = await aarch64.data(frame.header);
const selected = await Promise.all(ssa.map((fn, i) => aarch64.select(fn, [frame.header, ...others(i)].join('\\n'))));
const assembly = [data, ...selected].join('\\n');
const ran = await toolchain.runAssembly(assembly, inputs[0]);
return { ir: [frame.header, ...ssa].join('\\n'), assembly, diagnostics: [],
  log: ['verify ' + checked.ok, 'reference ' + JSON.stringify(reference.stdout), 'assembly ' + JSON.stringify(ran.stdout)] };`;

test('the pure pipeline reaches every stage and the toolchain through its callable folder', { skip }, async () => {
  const { model, seen } = compilerModel(PIPELINE);
  const runtime = createNatlangRuntime({ model: model.driver, codeEdits: 'deny' });
  const result = await runtime.run(() => compiler(SOURCE, 'c', 'O2', ['']), { services: { toolchain }, serviceDeclarations: { toolchain: toolchainDeclaration } });
  assert.deepEqual(result.log, ['verify true', 'reference "42\\n"', 'assembly "42\\n"']);
  const stages = seen.map(entry => entry.replace(/^(declare|data):.*/, '$1'));
  for (const stage of ['declare', 'lower:square', 'lower:main', 'mem2reg:square', 'mem2reg:main', 'data', 'select:square', 'select:main'])
    assert.ok(stages.includes(stage), `${stage} in ${seen.join(' ')}`);
  assert.equal((await toolchain.runAssembly(result.assembly)).stdout, '42\n');
});
