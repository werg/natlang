/**
 * The toolchain service: LLVM's verifier and JIT (through llvmlite) for IR, gcc's assembler and linker for machine
 * code. It stands for the outside world the compiler targets: it checks and runs, it never generates code.
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export type Check = { ok: boolean, error?: string };
export type Run = { ok: boolean, stdout: string, stderr: string, exitCode: number | null, ms: number, error?: string };

/** The Python with llvmlite: NATLANG_LLVM_PYTHON, else the development venv. */
export function llvmPython(): string {
  return process.env.NATLANG_LLVM_PYTHON ?? join(homedir(), '.local/share/natlang-dev/venv-llvm/bin/python');
}
export const toolchainAvailable = () => existsSync(llvmPython());

const script = fileURLToPath(new URL('../../compilers/toolchain.py', import.meta.url));
const helper = existsSync(script) ? script : fileURLToPath(new URL('./toolchain.py', import.meta.url));

function exec(command: string, args: string[], input: string, timeoutMs: number): Promise<Run> {
  return new Promise(resolve => {
    const started = performance.now();
    const child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '', timedOut = false;
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, timeoutMs);
    child.stdout.setEncoding('utf8').on('data', chunk => { if (stdout.length < 1 << 20) stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { if (stderr.length < 1 << 16) stderr += chunk; });
    child.stdin.on('error', () => {});
    child.stdin.end(input);
    child.on('error', error => { clearTimeout(timer); resolve({ ok: false, stdout, stderr, exitCode: null, ms: 0, error: error.message }); });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      const ms = Math.round(performance.now() - started);
      const error = timedOut ? `timed out after ${timeoutMs} ms` : signal ? `killed by ${signal}` : undefined;
      resolve({ ok: !error && code === 0, stdout, stderr, exitCode: code, ms, ...(error ? { error } : {}) });
    });
  });
}

async function scratch<T>(use: (directory: string) => Promise<T>): Promise<T> {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-cc-'));
  try { return await use(directory); } finally { await rm(directory, { recursive: true, force: true }); }
}

export const toolchain = {
  /** Parse and verify an LLVM IR module (LLVM 22, opaque pointers); the error is LLVM's own message. */
  async verify(ir: string): Promise<Check> {
    const result = await exec(llvmPython(), [helper, 'verify'], ir, 60_000);
    try { return JSON.parse(result.stdout) as Check; } catch { return { ok: false, error: result.error ?? result.stderr.trim() }; }
  },
  /** Run an IR module's main with this stdin (JIT); the program's stdout, stderr and exit code. */
  async runIR(ir: string, stdin = '', timeoutMs = 20_000): Promise<Run> {
    return scratch(async directory => {
      const path = join(directory, 'module.ll');
      await writeFile(path, ir);
      return exec(llvmPython(), [helper, 'run', path], stdin, timeoutMs);
    });
  },
  /** Assemble and link AArch64 assembly (with libc and libm), then run it with this stdin. */
  async runAssembly(assembly: string, stdin = '', timeoutMs = 20_000): Promise<Run> {
    return scratch(async directory => {
      await writeFile(join(directory, 'program.s'), assembly);
      const build = await exec('gcc', ['-o', join(directory, 'program'), join(directory, 'program.s'), '-lm'], '', 60_000);
      if (build.exitCode !== 0) return { ...build, ok: false, error: `assembler/linker: ${build.stderr.trim().slice(0, 2000)}` };
      return exec(join(directory, 'program'), [], stdin, timeoutMs);
    });
  },
};

/** What the model is told about the service. */
export const toolchainDeclaration = `/** Parse and verify an LLVM IR module (LLVM 22 syntax, opaque pointers \`ptr\`); declarations count as a module. */
export function verify(ir: string): Promise<{ ok: boolean, error?: string }>;
/** Run the module's main with this stdin; the program's own stdout and exit code. */
export function runIR(ir: string, stdin?: string): Promise<{ ok: boolean, stdout: string, stderr: string, exitCode: number | null, ms: number, error?: string }>;
/** Assemble and link AArch64 assembly (GNU as syntax, linked with libc and libm), then run it with this stdin. */
export function runAssembly(assembly: string, stdin?: string): Promise<{ ok: boolean, stdout: string, stderr: string, exitCode: number | null, ms: number, error?: string }>;`;

/** LLVM's own AArch64 assembly for a module: the validation scaffold (host only; not offered to the model). */
export async function llvmAssembly(ir: string): Promise<{ ok: boolean, assembly?: string, error?: string }> {
  const result = await exec(llvmPython(), [helper, 'assembly'], ir, 120_000);
  try { return JSON.parse(result.stdout); } catch { return { ok: false, error: result.error ?? result.stderr.trim() }; }
}

/** Reference builds for the benchmarks (not part of the compiler): gcc at an optimization level. */
export async function gccReference(source: string, level: string, stdin = '', timeoutMs = 60_000): Promise<Run> {
  return scratch(async directory => {
    await writeFile(join(directory, 'program.c'), source);
    const build = await exec('gcc', [level, '-o', join(directory, 'program'), join(directory, 'program.c'), '-lm'], '', 120_000);
    if (build.exitCode !== 0) return { ...build, ok: false, error: `gcc: ${build.stderr.trim().slice(0, 2000)}` };
    return exec(join(directory, 'program'), [], stdin, timeoutMs);
  });
}
