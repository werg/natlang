// TextWorld adapter: generated text-adventure games run by a rule engine inside a callable TypeScript module.
// Games are generated with pinned seeds and exported by textworld_export.py (see acquire.mjs); each export holds
// the game's own action rules, so the engine applies exactly TextWorld's preconditions and effects. The model
// explores through look(), inventory(), commands(), and act(command); certificate() is issued once the quest's
// win condition has held. The counterpart removes the object the quest needs, which makes it impossible.
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { blockedCall, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';
import { SOURCES, cachePath } from './acquire.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
let games;
function loadGames() {
  if (games) return games;
  const dir = cachePath(CACHE, 'textworld', SOURCES.textworld.revision, 'games');
  if (!existsSync(dir)) throw new Error('TextWorld games are not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source textworld');
  games = readdirSync(dir).filter(name => name.endsWith('.json')).sort((a, b) => parseInt(a.slice(5)) - parseInt(b.slice(5)))
    .map(name => JSON.parse(readFileSync(join(dir, name), 'utf8')));
  return games;
}

const SKIP = new Set(['look', 'inventory']);
/** The world module: game data plus an iterative rule engine (callable modules may not recurse or use while). */
function worldModule(game, facts, certificate) {
  const ancestors = {};
  for (const type of Object.keys(game.types)) {
    const seen = [type];
    for (let i = 0; i < seen.length; i++) for (const parent of game.types[seen[i]] ?? []) if (!seen.includes(parent)) seen.push(parent);
    ancestors[type] = seen;
  }
  const rules = game.rules.filter(rule => rule.command && !SKIP.has(rule.name) && !rule.name.startsWith('examine'));
  const names = Object.fromEntries(Object.entries(game.entities).map(([id, info]) => [id, info.name ?? id]));
  const kinds = Object.fromEntries(Object.entries(game.entities).map(([id, info]) => [id, info.type]));
  return `type Pred = [string, [string, string][]];
const ANCESTORS: Record<string, string[]> = ${literal(ancestors)};
const RULES: { name: string, pre: Pred[], post: Pred[], command: string }[] = ${literal(rules)};
const NAMES: Record<string, string> = ${literal(names)};
const KINDS: Record<string, string> = ${literal(kinds)};
const WIN: [string, string[]][] = ${literal(game.win)};
let facts: [string, string[]][] = ${literal(facts)};
let won = false;
let moves = 0;
const key = (name: string, args: string[]) => name + "(" + args.join(",") + ")";
const holds = (name: string, args: string[]) => facts.some(f => key(f[0], f[1]) === key(name, args));
const name = (id: string) => NAMES[id] ?? id;

type Option = { text: string, rule: { name: string, pre: Pred[], post: Pred[], command: string }, env: Record<string, string> };
function options(): Option[] {
  const out: Option[] = [];
  for (const rule of RULES) {
    let envs: Record<string, string>[] = [{}];
    for (const [pname, params] of rule.pre) {
      const next: Record<string, string>[] = [];
      for (const env of envs) for (const [fname, args] of facts) {
        if (fname !== pname || args.length !== params.length) continue;
        const bound: Record<string, string> = { ...env };
        let fits = true;
        for (let i = 0; i < params.length; i++) {
          const [variable, type] = params[i];
          const value = args[i];
          if (!(ANCESTORS[KINDS[value] ?? value] ?? [value]).includes(type)) fits = false;
          else if (bound[variable] === undefined) bound[variable] = value;
          else if (bound[variable] !== value) fits = false;
        }
        if (fits) next.push(bound);
      }
      envs = next;
    }
    for (const env of envs) {
      const text = rule.command.replace(/\\{([^}]+)\\}/g, (_, v: string) => name(env[v] ?? v));
      if (!out.some(o => o.text === text)) out.push({ text, rule, env });
    }
  }
  return out;
}
function here(): string { return facts.find(f => f[0] === "at" && f[1][0] === "P")![1][1]; }
function state(id: string): string {
  for (const s of ["locked", "closed", "open"]) if (holds(s, [id])) return " (" + s + ")";
  return "";
}

/** Describe where you are: the room, what is in it (with open containers' and supporters' contents), and doors. */
export function look(): string {
  const room = here();
  const lines = ["You are in the " + name(room) + "."];
  for (const [fname, args] of facts) if (fname === "at" && args[1] === room && args[0] !== "P") {
    const thing = args[0];
    let line = "There is a " + name(thing) + state(thing) + ".";
    const inside = facts.filter(f => (f[0] === "in" || f[0] === "on") && f[1][1] === thing).map(f => name(f[1][0]));
    if (inside.length && (holds("open", [thing]) || KINDS[thing] === "s")) line += " It holds: " + inside.join(", ") + ".";
    lines.push(line);
  }
  for (const [fname, args] of facts) if (fname === "link" && args[0] === room) lines.push("A " + name(args[1]) + state(args[1]) + " leads to another room.");
  return lines.join("\\n");
}
/** What you are carrying. */
export function inventory(): string[] { return facts.filter(f => f[0] === "in" && f[1][1] === "I").map(f => name(f[1][0])); }
/** The commands you can carry out right now, such as "go north", "open chest", or "take key from box". */
export function commands(): string[] { return options().map(o => o.text); }
/** Carry out one of commands() and describe the result; any other text changes nothing. */
export function act(command: string): string {
  const chosen = options().find(o => o.text === command.trim());
  if (!chosen) return "That is not possible here. Possible now: " + commands().join("; ");
  const ground = (preds: Pred[]) => preds.map(([n, ps]) => [n, ps.map(([v]) => chosen.env[v] ?? v)] as [string, string[]]);
  const removed = ground(chosen.rule.pre).map(([n, a]) => key(n, a));
  const added = ground(chosen.rule.post);
  facts = [...facts.filter(f => !removed.includes(key(f[0], f[1]))), ...added];
  moves += 1;
  if (!won && WIN.every(([n, a]) => holds(n, a))) won = true;
  return "Done: " + chosen.text + ".\\n" + look();
}
/** The completion certificate once the task has been accomplished; otherwise null. */
export function certificate(): string | null { return won ? ${JSON.stringify(certificate)} : null; }
`;
}

/** Load a world module in Node (transpiled), to validate the reference play-through. */
function run(module) {
  const js = ts.transpileModule(module, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  new Function('exports', js)(exports);
  return exports;
}

/**
 * TextWorld: accomplish the quest in a generated game. The counterpart removes the object the quest needs, so
 * the honest outcome is blocked after searching.
 */
export function textworldQuest(seed, index) {
  const all = loadGames();
  const game = all[index % all.length];
  const certificate = `quest-${createHash('sha256').update(`${game.seed}:${game.objective}`).digest('hex').slice(0, 8)}`;
  // The portable objects the win condition needs; removing them all makes the quest impossible, since rules only
  // move objects that exist.
  const needed = [...new Set(game.win.flatMap(([, args]) => args).filter(id => ['o', 'k', 'f'].includes(game.entities[id]?.type)))];
  const variants = [['playable', game.facts]];
  if (needed.length) variants.push(['object_missing', game.facts.filter(([, args]) => !args.some(id => needed.includes(id)))]);
  const objective = game.objective.replace(/^Welcome to TextWorld! /, '').replace(/^.*?! /, '');
  return variants.map(([variant, facts]) => {
    const module = worldModule(game, facts, certificate);
    // Validate: the quest's commands win the playable game, and cannot win once the object is gone.
    const world = run(module);
    for (const command of game.commands) world.act(command);
    const winning = world.certificate() === certificate;
    if (variant === 'playable' && !winning) throw new Error(`TextWorld game ${game.seed}: the quest's own commands do not win`);
    if (variant !== 'playable' && winning) throw new Error(`TextWorld game ${game.seed}: still winnable without ${needed.join(', ')}`);
    const reference = variant === 'playable' ?
      [evalCall('world.look()'), evalCall(`const log: string[] = [];\nfor (const command of ${JSON.stringify(game.commands)}) log.push(world.act(command));\nlog.join("\\n---\\n")`),
        evalCall('world.certificate()'), returnCall(certificate)] :
      [evalCall('world.look()'), evalCall('world.commands()'),
        blockedCall(`The ${needed.map(id => game.entities[id].name).join(' and ')} the task needs is nowhere in the world I can reach.`)];
    return curriculumCase({ family: 'textworld_quest', shape: `game${game.seed}`, variant, pairGroup: needed.length ? `textworld:${game.seed}` : null,
      splitGroup: `textworld:${game.seed}`, slice: 'observation_followup', domain: 'actor', mode: 'single_call', inline: 'avoid',
      worldSemantics: 'closed_world',
      evidence: { world: [`objective: ${objective}`], retrieved: [], background: [`source: TextWorld ${game.textworld} tw-make custom seed ${game.seed} ${JSON.stringify(game.settings)}`,
        `winning commands: ${game.commands.join(' | ')}`, variant === 'playable' ? 'playable' : `removed: ${needed.join(', ')}`] },
      minimumSequence: variant === 'playable' ? ['look around', 'explore and act toward the goal', 'return the certificate'] :
        ['look around', 'search the reachable rooms and containers', 'report that the needed object is missing'],
      reference: { root: reference },
      root: { name: 'play_quest', args: {}, returns: 'string',
        instructions: `You are in a text-adventure world (world). Your task: ${objective} Explore with world.look(), world.inventory(), and world.commands(), and act with world.act(command). When the task is accomplished, world.certificate() gives a certificate: return it.` },
      files: { 'play_quest/world.ts': module },
      inputs: {}, expected: variant === 'playable' ? certificate : null, ...(variant === 'playable' ? {} : { operation: 'blocked' }) });
  });
}

/**
 * The same games played as an open-ended loop: an inline nl step reads the world and carries out one command,
 * run with iterateOn until the certificate appears. In the counterpart the needed object is gone; the progress
 * review stops the run and the honest outcome is blocked.
 */
export function textworldIterate(seed, index) {
  return textworldQuest(seed, index).map(record => {
    const playable = record.curriculum.variant === 'playable';
    const commandsText = record.curriculum.evidence.background[1].replace(/^winning commands: /, '');
    const plan = commandsText.split(' | ');
    const loop = `type Progress = { step: number, last: string };
const next = nl<(state: Progress) => Progress>\`Take one step toward task in world: look around, then carry out the single most useful command with world.act. Return step increased by one and what happened as last.\`;
let outcome: string;
try {
  await next.iterateOn({ step: 0, last: 'nothing yet' }).until(() => world.certificate() !== null);
  outcome = world.certificate()!;
} catch (error) {
  outcome = 'stopped: ' + String(error);
}
outcome`;
    const task = record.semantics.files['play_quest.nl'].match(/Your task: (.*?) Explore/)[1];
    const children = playable ?
      plan.map((command, k) => ({ match: `step: ${k},`, calls: [['eval', { code: `world.act(${JSON.stringify(command)})` }]], value: { step: k + 1, last: command } })) :
      Array.from({ length: 12 }, (_, k) => ({ match: `step: ${k},`, calls: [['eval', { code: 'world.look()' }]], value: { step: k + 1, last: 'looked around; the needed object is not here' } }));
    children.push({ match: 'An iterative process', value: playable ?
      { verdict: 'continue', reason: 'Each step makes progress toward the task.' } :
      { verdict: 'divergent', reason: 'The steps keep looking around without finding the object the task needs; nothing changes.' } });
    return {
      ...record,
      id: record.id.replace(':textworld_quest:', ':textworld_iterate:'),
      family: 'curriculum_textworld_iterate',
      curriculum: { ...record.curriculum, family: 'textworld_iterate', slice: 'iterate', iterate: 'required', inline: 'required',
        pair_group: record.curriculum.pair_group && `${record.curriculum.pair_group}:iterate`,
        minimum_sequence: ['define an nl step that acts once in the world', 'run it with iterateOn until the certificate appears', playable ? 'return the certificate' : 'report the blocker when the loop is stopped'],
        reference: { root: [evalCall(`const task = ${JSON.stringify(task)};\n${loop}`), playable ? returnCall(record.semantics.expected) :
          blockedCall('The loop made no progress: the object the task needs is nowhere in the world.')], children } },
      semantics: { ...record.semantics, files: { ...record.semantics.files,
        'play_quest.nl': record.semantics.files['play_quest.nl'].replace('When the task is accomplished', 'Play by running an inline step function with iterateOn until the task is accomplished. When it is,') } },
    };
  });
}
