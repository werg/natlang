/**
 * The agent's surface as pi presents it: the system prompt text and the tool schemas a request carries. Training
 * records built from other agents' trajectories (plans/neuralese/HARNESS_BENCH.md) take both from here, so the student
 * learns the harness it will run in, and a change to the prompt or the tools reaches the next data build.
 */
import { getSystemMessageText } from '@earendil-works/pi-ai';
import { codingTools } from './extensions/coding-tools/index.ts';
import { companion } from './extensions/companion/index.ts';
import { buildSystemPromptSections, CONTRIBUTIONS } from './extensions/pi-prompt/sections.ts';

export type AgentSurface = {
  /** The system message as the provider renders it (pi-ai getSystemMessageText). */
  system: string;
  /** The request's tools, as JSON Schema. */
  tools: { name: string; description: string; parameters: unknown }[];
};

/**
 * The surface for an agent working in `cwd` with the coding tools, and with the companion's `recall` when
 * `companion` is set. `packageDir` is pi's package directory as the docs section names it.
 */
export function agentSurface(options: { cwd: string; companion?: boolean; packageDir?: string }): AgentSurface {
  const registrations = [...(codingTools(undefined as never).tools ?? []),
    ...(options.companion ? companion(undefined as never, { harness: () => undefined as never }).tools ?? [] : [])];
  const tools = registrations.map(tool => ({ name: tool.name, description: tool.description,
    parameters: JSON.parse(JSON.stringify(tool.parameters)) as unknown }));
  const selectedTools = tools.map(tool => tool.name);
  const toolSnippets: Record<string, string> = {}, toolGuidelines: Record<string, string[]> = {};
  for (const name of selectedTools) {
    const contribution = CONTRIBUTIONS[name];
    if (!contribution) continue;
    toolSnippets[name] = contribution.snippet;
    toolGuidelines[name] = [...contribution.guidelines];
  }
  const sections = buildSystemPromptSections({ cwd: options.cwd, selectedTools, toolSnippets, toolGuidelines,
    packageDir: options.packageDir ?? '/opt/pi', contextFiles: [], skills: [] });
  return { system: getSystemMessageText({ role: 'system', content: [], sections } as never), tools };
}
