/**
 * The `resources` of pi's coding agent for a directory: project context files (AGENTS.md / CLAUDE.md from the agent
 * directory and up the tree) and skills (SKILL.md and skill files from the agent and project skill directories and
 * configured paths), loaded as pi's coding agent loads them (core/resource-loader.ts, core/skills.ts).
 *
 * One deviation: pi skips skill files that .gitignore, .ignore or .fdignore files exclude; the port does not read
 * ignore files (pi's `ignore` package is not a dependency here).
 */
import { existsSync, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, isAbsolute, join, resolve, sep } from 'node:path';
import { parse as parseYaml } from 'yaml';

export const CONFIG_DIR_NAME = '.pi';

export type ContextFile = { path: string; content: string };
export type Skill = { name: string; description: string; filePath: string; baseDir: string; disableModelInvocation: boolean };

const stripBom = (text: string) => text.startsWith('﻿') ? text.slice(1) : text;
const expandTilde = (path: string) => path === '~' ? homedir() : path.startsWith('~/') ? join(homedir(), path.slice(2)) : path;
const resolvePath = (input: string, base = process.cwd()) => {
  const path = expandTilde(input.trim());
  return isAbsolute(path) ? resolve(path) : resolve(base, path);
};
const canonical = (path: string) => { try { return realpathSync(path); } catch { return path; } };

/** pi's agent directory: $PI_CODING_AGENT_DIR, else ~/.pi/agent. */
export function agentDir(): string {
  const env = process.env.PI_CODING_AGENT_DIR;
  return env ? expandTilde(env) : join(homedir(), CONFIG_DIR_NAME, 'agent');
}

function contextFileIn(dir: string): ContextFile | null {
  for (const name of ['AGENTS.override.md', 'AGENTS.md', 'AGENTS.MD', 'CLAUDE.md', 'CLAUDE.MD']) {
    const path = join(dir, name);
    if (!existsSync(path)) continue;
    try {
      if (!statSync(path).isFile()) continue;
      return { path, content: stripBom(readFileSync(path, 'utf-8')) };
    } catch { /* unreadable: pi warns and goes on */ }
  }
  return null;
}

function gitPaths(cwd: string): { repoDir: string; commonGitDir: string } | null {
  for (let dir = cwd; ; dir = dirname(dir)) {
    const git = join(dir, '.git');
    if (existsSync(git)) {
      try {
        const stat = statSync(git);
        if (stat.isFile()) {
          const content = readFileSync(git, 'utf8').trim();
          if (content.startsWith('gitdir: ')) {
            const gitDir = resolve(dir, content.slice(8).trim());
            if (!existsSync(join(gitDir, 'HEAD'))) return null;
            const commondir = join(gitDir, 'commondir');
            return { repoDir: dir, commonGitDir: existsSync(commondir) ? resolve(gitDir, readFileSync(commondir, 'utf8').trim()) : gitDir };
          }
        } else if (stat.isDirectory()) return existsSync(join(git, 'HEAD')) ? { repoDir: dir, commonGitDir: git } : null;
      } catch { return null; }
    }
    if (dirname(dir) === dir) return null;
  }
}

/** The main repository's context file that a nested linked worktree's own copy shadows, canonicalized. */
function shadowedContextFile(cwd: string): string | undefined {
  const paths = gitPaths(cwd);
  if (!paths) return undefined;
  const common = canonical(paths.commonGitDir), worktree = canonical(paths.repoDir), main = dirname(common);
  if (!worktree.startsWith(`${main}${sep}`)) return undefined;
  if (canonical(join(main, '.git')) !== common) return undefined;
  const own = contextFileIn(worktree);
  return own ? join(main, basename(own.path)) : undefined;
}

/** pi's project context files for cwd: the agent directory's, then each ancestor's from the root down to cwd. */
export function loadProjectContextFiles(cwd: string, agent = agentDir()): ContextFile[] {
  const files: ContextFile[] = [], seen = new Set<string>();
  const global = contextFileIn(resolvePath(agent));
  if (global) { files.push(global); seen.add(global.path); }
  const ancestors: ContextFile[] = [];
  const resolvedCwd = resolvePath(cwd);
  const shadowed = shadowedContextFile(resolvedCwd);
  for (let dir = resolvedCwd; ; dir = dirname(dir)) {
    const file = contextFileIn(dir);
    const isShadowed = shadowed !== undefined && canonical(file?.path ?? '') === shadowed;
    if (file && !isShadowed && !seen.has(file.path)) { ancestors.unshift(file); seen.add(file.path); }
    if (dirname(dir) === dir) break;
  }
  return [...files, ...ancestors];
}

function frontmatter(content: string): Record<string, unknown> {
  const normalized = stripBom(content).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  if (!normalized.startsWith('---')) return {};
  const end = normalized.indexOf('\n---', 3);
  if (end === -1) return {};
  return (parseYaml(normalized.slice(4, end)) ?? {}) as Record<string, unknown>;
}

function skillFromFile(filePath: string): Skill | null {
  let parsed: Record<string, unknown>;
  try { parsed = frontmatter(readFileSync(filePath, 'utf-8')); } catch { return null; }
  const description = parsed.description;
  if (typeof description !== 'string' || description.trim() === '') return null;
  const baseDir = dirname(filePath);
  const name = (typeof parsed.name === 'string' && parsed.name) || basename(baseDir);
  return { name, description, filePath, baseDir, disableModelInvocation: parsed['disable-model-invocation'] === true };
}

/** pi's discovery: a directory with SKILL.md is a skill root; otherwise direct .md children (at the root only) and subdirectories. */
function skillsInDir(dir: string, includeRootFiles: boolean): Skill[] {
  if (!existsSync(dir)) return [];
  const skills: Skill[] = [];
  try {
    const entries = readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name !== 'SKILL.md') continue;
      const path = join(dir, entry.name);
      let isFile = entry.isFile();
      if (entry.isSymbolicLink()) { try { isFile = statSync(path).isFile(); } catch { continue; } }
      if (!isFile) continue;
      const skill = skillFromFile(path);
      return skill ? [skill] : [];
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
      const path = join(dir, entry.name);
      let isDirectory = entry.isDirectory(), isFile = entry.isFile();
      if (entry.isSymbolicLink()) {
        try { const stats = statSync(path); isDirectory = stats.isDirectory(); isFile = stats.isFile(); } catch { continue; }
      }
      if (isDirectory) { skills.push(...skillsInDir(path, false)); continue; }
      if (!isFile || !includeRootFiles || !entry.name.endsWith('.md')) continue;
      const skill = skillFromFile(path);
      if (skill) skills.push(skill);
    }
  } catch { /* unreadable directory: none */ }
  return skills;
}

/** pi's skills for cwd: the agent's and the project's skill directories, then `skillPaths`; the first of a name wins. */
export function loadSkills(cwd: string, options: { agentDir?: string; skillPaths?: string[]; includeDefaults?: boolean } = {}): Skill[] {
  const resolvedCwd = resolvePath(cwd), agent = resolvePath(options.agentDir ?? agentDir());
  const byName = new Map<string, Skill>(), realPaths = new Set<string>();
  const add = (skills: Skill[]) => {
    for (const skill of skills) {
      const real = canonical(skill.filePath);
      if (realPaths.has(real) || byName.has(skill.name)) continue;
      byName.set(skill.name, skill);
      realPaths.add(real);
    }
  };
  if (options.includeDefaults ?? true) {
    add(skillsInDir(join(agent, 'skills'), true));
    add(skillsInDir(resolve(resolvedCwd, CONFIG_DIR_NAME, 'skills'), true));
  }
  for (const raw of options.skillPaths ?? []) {
    const path = resolvePath(raw, resolvedCwd);
    if (!existsSync(path)) continue;
    try {
      const stats = statSync(path);
      if (stats.isDirectory()) add(skillsInDir(path, true));
      else if (stats.isFile() && path.endsWith('.md')) { const skill = skillFromFile(path); if (skill) add([skill]); }
    } catch { /* unreadable: pi records a diagnostic */ }
  }
  return [...byName.values()];
}

const escapeXml = (text: string) => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');

/** pi's skills prompt (Agent Skills XML); `fileReadTool` names the tool that loads skill files. */
export function formatSkillsForPrompt(skills: Skill[], fileReadTool: 'read' | 'bash' | 'indirect' = 'read'): string {
  const visible = skills.filter(skill => !skill.disableModelInvocation);
  if (visible.length === 0) return '';
  const lines = [
    '\n\nThe following skills provide specialized instructions for specific tasks.',
    fileReadTool === 'read' ? "Use the read tool to load a skill's file when the task matches its description." :
      fileReadTool === 'bash' ? "Use bash to load a skill's file when the task matches its description." :
        "Load a skill's file when the task matches its description.",
    'When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.',
    '',
    '<available_skills>',
  ];
  for (const skill of visible) {
    lines.push('  <skill>', `    <name>${escapeXml(skill.name)}</name>`, `    <description>${escapeXml(skill.description)}</description>`,
      `    <location>${escapeXml(skill.filePath)}</location>`, '  </skill>');
  }
  lines.push('</available_skills>');
  return lines.join('\n');
}

/** Context files and skills of each directory, loaded once per directory, as pi loads them at startup. */
export class Resources {
  readonly #loaded = new Map<string, { contextFiles: ContextFile[]; skills: Skill[] }>();
  readonly options: { agentDir?: string; skillPaths?: string[] };
  constructor(options: { agentDir?: string; skillPaths?: string[] } = {}) { this.options = options; }
  for(cwd: string): { contextFiles: ContextFile[]; skills: Skill[] } {
    let found = this.#loaded.get(cwd);
    if (!found) {
      found = { contextFiles: loadProjectContextFiles(cwd, this.options.agentDir), skills: loadSkills(cwd, { ...this.options, includeDefaults: true }) };
      this.#loaded.set(cwd, found);
    }
    return found;
  }
}
