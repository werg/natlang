/**
 * pi's system prompt as one extension, "pi-prompt" (coding-agent experimental/durable/prompt.ts): the sections of
 * pi's `buildSystemPromptSections()` (core/system-prompt.ts) for the request's tools and the conversation's directory,
 * verbatim, including the bash guideline about PI_* environment variables (owner decision 14). Renderers must be
 * deterministic: any change in rendered text appends a system delta and breaks provider caches (§7.4).
 */
import type { Extension, PromptInput, PromptSection } from '../../vendor/durable/src/harness/types.ts';
import { joinPath } from '../../host/paths.ts';

export type ContextFile = { path: string; content: string };
export type Skill = { name: string; description: string; filePath: string; baseDir: string; disableModelInvocation: boolean };
/**
 * Where a directory's project context files and skills come from: the Node host loads them from the file system as pi
 * does (host/resources.ts); a host without one has none.
 */
export type PromptResources = { for(cwd: string): { contextFiles: ContextFile[]; skills: Skill[] } };

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

/** The tools' contributions to the prompt (core/tools/{read,bash,edit,write}.ts). */
export const CONTRIBUTIONS: Record<string, { snippet: string; guidelines: readonly string[] }> = {
  read: { snippet: 'Read file contents', guidelines: ['Use read to examine files instead of cat or sed.'] },
  bash: { snippet: 'Execute bash commands (ls, grep, find, etc.)', guidelines: ['You can inspect PI_* environment variables for current model and session details.'] },
  edit: {
    snippet: 'Make precise file edits with exact text replacement, including multiple disjoint edits in one call',
    guidelines: [
      'Use edit for precise changes (edits[].oldText must match exactly)',
      'When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls',
      'Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.',
      'Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.',
    ],
  },
  write: { snippet: 'Create or overwrite files', guidelines: ['Use write only for new files or complete rewrites.'] },
};

/** pi's section order; sections without content are omitted. */
const KEYS = ['preamble', 'tools', 'rules', 'docs', 'project_context', 'skills', 'cwd'] as const;

export type PromptBuild = {
  cwd: string;
  selectedTools: string[];
  toolSnippets: Record<string, string>;
  toolGuidelines: Record<string, string[]>;
  contextFiles: ContextFile[];
  skills: Skill[];
  /** pi's package directory, for the docs section. */
  packageDir: string;
};

function renderProjectContext(contextFiles: ContextFile[]): string {
  return ['Project-specific instructions and guidelines:',
    ...contextFiles.map(({ path, content }) => `<project_instructions path="${path}">\n${content}\n</project_instructions>`)].join('\n\n');
}

function buildRules(selectedTools: string[], toolGuidelines: Record<string, string[]>): string {
  const rules: string[] = [], seen = new Set<string>();
  const add = (rule: string) => { const normalized = rule.trim(); if (!normalized || seen.has(normalized)) return; seen.add(normalized); rules.push(normalized); };
  const hasBash = selectedTools.includes('bash'), hasPowerShell = selectedTools.includes('powershell');
  if ((hasBash || hasPowerShell) && !selectedTools.includes('grep') && !selectedTools.includes('find') && !selectedTools.includes('ls')) {
    if (hasBash && hasPowerShell) add('Use bash or PowerShell for file operations like listing, searching, and finding files');
    else if (hasPowerShell) add('Use PowerShell for file operations like listing, searching, and finding files');
    else add('Use bash for file operations like ls, rg, find');
  }
  for (const name of selectedTools) for (const rule of toolGuidelines[name] ?? []) add(rule);
  add('Be concise in your responses');
  add('Show file paths clearly when working with files');
  return rules.map(rule => `- ${rule}`).join('\n');
}

/** pi's buildSystemPromptSections for the default prompt (no custom prompt, addendum or extra sections). */
export function buildSystemPromptSections(input: PromptBuild): Record<string, string> {
  const { selectedTools, toolSnippets, toolGuidelines, cwd, contextFiles, skills } = input;
  const readme = joinPath(input.packageDir, 'README.md'), docs = joinPath(input.packageDir, 'docs'),
    examples = joinPath(input.packageDir, 'examples');
  const sections: Record<string, string> = {};
  sections.preamble = 'You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.';
  const visible = selectedTools.filter(name => !!toolSnippets[name]);
  const tools = visible.length > 0 ? visible.map(name => `- ${name}: ${toolSnippets[name]}`).join('\n') : '(none)';
  sections.tools = `${tools}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.`;
  sections.rules = buildRules(selectedTools, toolGuidelines);
  sections.docs = `Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: ${readme}
- Additional docs: ${docs}
- Examples: ${examples} (extensions, custom tools, SDK)
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md), MCP servers (docs/mcp.md), codemode scripts and non-LLM models such as classifiers and image models (docs/codemode.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)`;
  if (contextFiles.length > 0) sections.project_context = renderProjectContext(contextFiles);
  const reader = (['read', 'bash'] as const).find(tool => selectedTools.includes(tool));
  if (reader && skills.length > 0) {
    const prompt = formatSkillsForPrompt(skills, reader).trim();
    if (prompt) sections.skills = prompt;
  }
  sections.cwd = cwd.replace(/\\/g, '/');
  const result: Record<string, string> = { preamble: sections.preamble };
  for (const [name, content] of Object.entries(sections)) if (name !== 'preamble') result[name] = `<${name}>\n${content}\n</${name}>`;
  return result;
}

export type PiPromptOptions = {
  /** The directory used when neither the environment nor the agent names one. */
  cwd: string;
  /** pi's package directory, as the docs section names it. */
  packageDir: string;
  /** Each directory's context files and skills (default: none). */
  resources?: PromptResources;
};

const NO_RESOURCES: PromptResources = { for: () => ({ contextFiles: [], skills: [] }) };

/** pi's system prompt as an extension of seven untagged sections (each carries its own tag). */
export function piPrompt(options: PiPromptOptions): Extension {
  const resources = options.resources ?? NO_RESOURCES;
  const packageDir = options.packageDir;
  // The sections of one request render from one build.
  const built = new WeakMap<PromptInput, Record<string, string>>();
  const build = (input: PromptInput): Record<string, string> => {
    let sections = built.get(input);
    if (!sections) {
      const cwd = input.env?.cwd ?? input.agent.cwd ?? options.cwd;
      const selectedTools = input.agent.tools.map(tool => tool.name);
      const toolSnippets: Record<string, string> = {}, toolGuidelines: Record<string, string[]> = {};
      for (const name of selectedTools) {
        const contribution = CONTRIBUTIONS[name];
        if (!contribution) continue;
        toolSnippets[name] = contribution.snippet;
        toolGuidelines[name] = [...contribution.guidelines];
      }
      sections = buildSystemPromptSections({ cwd, selectedTools, toolSnippets, toolGuidelines, packageDir, ...resources.for(cwd) });
      built.set(input, sections);
    }
    return sections;
  };
  const sections: PromptSection[] = KEYS.map(key => ({ key, tag: false, render: (input: PromptInput) => build(input)[key] }));
  return { name: 'pi-prompt', sections };
}
