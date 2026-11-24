/**
 * `natlang run applications/program-inventor -- CONFIG.json`
 *
 * CONFIG: { "out": DIR, "count": N, "builder": "claude" | "codex", "builderModel": ID, "teacher": TEACHER_SPEC }
 * Models of the judgments come from the runtime's models named `inventor`, `judge` (another family than the inventor)
 * and `reviewer` (another family than the builder). Writes DIR/<slug>/ submissions, DIR/accepted.jsonl (the ledger the
 * atlas reads) and DIR/round.json. Run on the DGX through the memory ledger.
 */
import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { inventRound, type InventorServices } from './invent.js';
import type { Atlas, AtlasEntry, IdeaCard, LiveRun, ProgramView, Qualification, Requirement } from './types.js';

type Config = { out: string; count: number; builder: 'claude' | 'codex'; builderModel: string; teacher: Record<string, unknown> };

const REPO = resolve(new URL('.', import.meta.url).pathname, '../..');
const PYTHON = join(REPO, '.venv-neuralese/bin/python');
const ENV = { ...process.env, PYTHONPATH: `${REPO}/training/neuralese:${REPO}` };

function run(command: string, args: string[], cwd = REPO): string {
  return execFileSync(command, args, { cwd, env: ENV, encoding: 'utf8', maxBuffer: 1 << 28 });
}

function filesUnder(dir: string): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (path: string) => {
    for (const name of readdirSync(path)) {
      const full = join(path, name);
      if (name.startsWith('.') || name === 'node_modules') continue;
      if (statSync(full).isDirectory()) walk(full);
      else out[relative(dir, full)] = readFileSync(full, 'utf8');
    }
  };
  walk(dir);
  return out;
}

function services(config: Config): InventorServices {
  const ledger = join(config.out, 'accepted.jsonl');
  const atlasJson = join(config.out, 'atlas.json');
  const brief = readFileSync(join(REPO, 'applications/program-inventor/BUILD_BRIEF.md'), 'utf8');
  const requirements = JSON.parse(readFileSync(join(REPO, 'applications/program-inventor/requirements.json'), 'utf8')) as Requirement[];
  const agent = (directory: string, prompt: string) => config.builder === 'codex'
    ? run('codex', ['exec', '-m', config.builderModel, '-C', directory, '--dangerously-bypass-approvals-and-sandbox', prompt])
    : run('claude', ['-p', '--model', config.builderModel, '--permission-mode', 'bypassPermissions', '--add-dir', directory, prompt], directory);
  return {
    requirements,
    async atlas(): Promise<Atlas> {
      const markdown = join(config.out, 'atlas.md');
      run(PYTHON, ['scripts/rl_tasks/program_atlas.py', 'build', '--accepted', ledger, '-o', atlasJson, '--markdown', markdown]);
      const data = JSON.parse(readFileSync(atlasJson, 'utf8')) as { entries: AtlasEntry[]; commit: string };
      return { text: readFileSync(markdown, 'utf8'), entries: data.entries, commit: data.commit };
    },
    async near(idea: IdeaCard): Promise<AtlasEntry[]> {
      const path = join(config.out, `.idea-${idea.slug}.json`);
      writeFileSync(path, JSON.stringify(idea));
      const nearest = JSON.parse(run(PYTHON, ['scripts/rl_tasks/program_atlas.py', 'near', atlasJson, path, '-k', '12'])) as AtlasEntry[];
      const accepted = (JSON.parse(readFileSync(atlasJson, 'utf8')).entries as AtlasEntry[]).filter(entry => entry.kind === 'invented');
      return [...nearest, ...accepted.filter(entry => !nearest.some(near => near.id === entry.id))];
    },
    async build(idea: IdeaCard): Promise<string> {
      const directory = join(config.out, idea.slug);
      mkdirSync(directory, { recursive: true });
      writeFileSync(join(directory, '.idea.json'), JSON.stringify(idea, null, 2));
      writeFileSync(join(directory, '.requirements.json'), JSON.stringify(requirements, null, 2));
      agent(directory, `${brief}\n\nThe idea card is ${directory}/.idea.json and the requirements are ${directory}/.requirements.json. `
        + `Write the submission in ${directory}. The repository is ${REPO}.`);
      return directory;
    },
    async revise(submission: string, problems: string[]): Promise<string> {
      agent(submission, `${brief}\n\nYour submission in ${submission} has these problems; fix every one, then rerun the checks:\n- ${problems.join('\n- ')}`);
      return submission;
    },
    async qualify(submission: string): Promise<Qualification> {
      try { run(PYTHON, ['scripts/rl_tasks/sources/invented.py', 'qualify', submission]); } catch { /* the report says why */ }
      const report = JSON.parse(readFileSync(join(submission, 'qualification/report.json'), 'utf8'));
      return { problems: report.problems, metrics: JSON.stringify(report.metrics), cases: join(submission, 'qualification/cases.jsonl') };
    },
    async view(submission: string, idea: IdeaCard, metrics: string): Promise<ProgramView> {
      const data = JSON.parse(readFileSync(join(submission, 'submission.json'), 'utf8'));
      return { idea, files: filesUnder(join(submission, 'program')), instance: JSON.stringify(data.instances[0].inputs), metrics };
    },
    async liveRun(submission: string): Promise<LiveRun[]> {
      const work = join(submission, 'live');
      const cases = join(submission, 'qualification/cases.jsonl');
      try {
        run(PYTHON, ['scripts/neuralese_data/teacher_gen.py', '--work', work, '--pool', cases, '--teacher', JSON.stringify(config.teacher), '--attempts', '1']);
      } catch { /* scored episodes are what count */ }
      const rows = readFileSync(cases, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
      // teacher_gen's manifest: one record per scored attempt, {task_id, teacher, episodes: [{outcome, reward}], rollouts}
      const manifest = join(work, 'manifest.jsonl');
      const records = existsSync(manifest)
        ? readFileSync(manifest, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)) as
          { task_id: string; teacher: unknown; episodes: { outcome: string }[]; rollouts: string }[]
        : [];
      return rows.map(row => {
        const mine = records.filter(record => record.task_id === row.id);
        const passed = mine.some(record => record.episodes.some(episode => episode.outcome === 'task_success'));
        return { instance: JSON.stringify(row.semantics.inputs), expected: JSON.stringify(row.semantics.expected), passed,
          answer: 'null', issues: mine.map(record => `${JSON.stringify(record.teacher)}: ${record.episodes.map(e => e.outcome).join(', ')} (${record.rollouts})`) };
      });
    },
    async record(idea: IdeaCard, submission: string): Promise<void> {
      appendFileSync(ledger, JSON.stringify({ idea, submission, at: new Date().toISOString() }) + '\n');
    },
  };
}

export async function main(context: TargetContext): Promise<number> {
  const path = context.args[0];
  if (!path) throw new Error('usage: natlang run applications/program-inventor -- CONFIG.json');
  const config = JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8')) as Config;
  mkdirSync(config.out, { recursive: true });
  const results = await context.runtime.run(() => inventRound(services(config), config.count));
  writeFileSync(join(config.out, 'round.json'), JSON.stringify(results, null, 2));
  context.io.output.write(`${JSON.stringify(results, null, 2)}\n`);
  return results.some(result => result.status === 'accepted') ? 0 : 1;
}
