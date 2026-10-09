/**
 * `natlang run applications/games -- economy|combat|village [--turns N] [--crisp]`: play a demo world for N turns
 * (default 3) and print each turn's narration. With --crisp the rules run as their crisp references and only the
 * actors' own choices are made by the model.
 */
import type { TargetContext } from '@natlang/node';
import { Session, crispSettings, createArena, createEconomy, createVillage, defaultSettings, playTurn, type World } from './index.js';

const scenes: Record<string, () => World> = {
  economy: () => createEconomy([
    { id: 'alice', cash: 10, goods: { apple: 0, bread: 0 }, offers: {} },
    { id: 'bob', cash: 0, goods: { apple: 2, bread: 0 }, offers: { apple: 3 } },
    { id: 'cara', cash: 0, goods: { apple: 0, bread: 2 }, offers: { bread: 2 } }], { seed: 7 }),
  combat: () => createArena([{ id: 'a', x: 1, hp: 5 }, { id: 'b', x: 5, hp: 5 }]),
  village: () => createVillage([{ id: 'innkeeper', inventory: { key: 1 } }, { id: 'guard', inventory: {} }]),
};
const lines = ['May I have a key?', 'Will you help me find the map?', 'Thank you.'];

export async function main(context: TargetContext): Promise<number> {
  const [name, ...args] = context.args;
  const scene = name ? scenes[name] : undefined;
  if (!scene) { context.io.error.write(`usage: ${Object.keys(scenes).join('|')} [--turns N] [--crisp]\n`); return 2; }
  const turns = Number(args[args.indexOf('--turns') + 1] || 3);
  const session = new Session(scene(), args.includes('--crisp') ? crispSettings : defaultSettings,
    (sceneToPlay, settings) => context.runtime.run(() => playTurn(sceneToPlay, settings)));
  for (let i = 0; i < turns; i++) {
    const report = await session.turn(session.state.kind === 'npc' ? { actor: 'innkeeper', event: { from: 'guest', text: lines[i % lines.length]! } } : undefined);
    context.io.output.write(`${report.ok ? report.narration : `turn failed: ${report.problem}`}\n`);
    for (const line of report.log) context.io.output.write(`  ${line}\n`);
  }
  return 0;
}
