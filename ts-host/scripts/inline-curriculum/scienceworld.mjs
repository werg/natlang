// ScienceWorld adapter: the simulator runs in its own process and is the program's `world` service (see
// src/teacher/world-bridge.ts); a case is accepted when the task's score reaches 100. Tasks, descriptions, and gold
// action paths come from the pinned package (acquire.mjs --source scienceworld); the gold path is the reference.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { curriculumCase, evalCall, returnCall } from './lib.mjs';
import { SOURCES, cachePath } from './acquire.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
let tasks;
function loadTasks() {
  if (tasks) return tasks;
  const path = cachePath(CACHE, 'scienceworld', SOURCES.scienceworld.revision, 'tasks.json');
  if (!existsSync(path)) throw new Error('ScienceWorld tasks are not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source scienceworld');
  tasks = JSON.parse(readFileSync(path, 'utf8')).tasks.filter(task => task.gold.length && !String(task.gold[0]).startsWith('ERROR'));
  return tasks;
}

const API = 'world is a live science simulator. await world.look() describes where you are; await world.inventory() lists what you carry; ' +
  'await world.act(command) carries out one command (for example "open door to kitchen", "pick up thermometer", "focus on water") and returns ' +
  '{ observation, score, done }; await world.actions() lists the command templates and the objects you can name; await world.score() reports progress. ' +
  'Focus only on the thing the task names: focusing on the wrong thing fails the task.';

/** A ScienceWorld task, played through the world service until its score reaches 100. */
export function scienceworldTask(seed, index) {
  const all = loadTasks();
  const task = all[index % all.length];
  const reference = [evalCall(`const log: string[] = [];
for (const command of ${JSON.stringify(task.gold)}) {
  const step = await world.act(command);
  log.push(command + ' -> ' + step.observation.slice(0, 80));
}
const progress = await world.score();
({ progress, last: log.slice(-3) })`), returnCall('The task is complete.')];
  return [curriculumCase({ family: 'scienceworld_task', shape: `${task.task}_v${task.variation}`, variant: 'easy', splitGroup: `scienceworld:${task.task}`,
    split: task.variation === 2 ? 'test' : 'train', slice: 'observation_followup', domain: 'actor', mode: 'single_call', inline: 'avoid',
    worldSemantics: 'closed_world',
    evidence: { world: [task.description], retrieved: [], background: [`source: ScienceWorld ${SOURCES.scienceworld.revision} ${task.task} variation ${task.variation}`, `gold path: ${task.gold.length} commands`] },
    minimumSequence: ['look around', 'find and use what the task needs', 'reach a score of 100'],
    reference: { root: reference },
    root: { name: 'science_task', args: {}, returns: 'string', instructions: `${task.description}\n\n${API} Work until world.score() shows a score of 100, then reply with a one-sentence summary.` },
    inputs: {}, expected: null })].map(record => ({ ...record, semantics: { ...record.semantics,
      world: { kind: 'scienceworld', task: task.task, variation: task.variation, simplifications: task.simplifications } } }));
}
