// Short, balanced ChessT scenarios for adversarial self-play. A seeded generator places both kings and a few pieces
// per side (equal material) on an empty board; the pinned rules kernel validates each state, and random-vs-random
// playouts keep only positions that usually finish under the actual victory rules within the arena's decision cap
// and that both colours win sometimes (so both seats have something to learn). No cap-derived rewards.
import { createChesstGame } from '../../dist/self-play/chesst.js';

function rng(seed) {  // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const PIECES = ['queen', 'rook', 'knight', 'sage'];

// Kings toward the centre, an equal set of one or two piece types per side: positions where play meets quickly.
function position(random) {
  const board = Array.from({ length: 8 }, () => Array(8).fill(null));
  const taken = new Set();
  const place = (piece, r0, r1) => {
    for (;;) {
      const r = r0 + Math.floor(random() * (r1 - r0 + 1)), c = 1 + Math.floor(random() * 6);
      if (!taken.has(r * 8 + c)) { taken.add(r * 8 + c); board[r][c] = piece; return; }
    }
  };
  place({ type: 'king', color: 'gold', hasMoved: true }, 4, 6);
  place({ type: 'king', color: 'blue', hasMoved: true }, 1, 3);
  for (let i = 0, n = 1 + Math.floor(random() * 2); i < n; i++) {
    const type = PIECES[Math.floor(random() * PIECES.length)];
    place({ type, color: 'gold', hasMoved: true }, 2, 7);
    place({ type, color: 'blue', hasMoved: true }, 0, 5);
  }
  const first = random() < 0.5 ? 'gold' : 'blue';
  return { board, activePlayer: first, nominalPlayer: first, decisionPlayer: first, phase: 'CHOOSE_ACTION',
    pending: null, captured: { gold: [], blue: [] }, sorcererCooldown: { gold: 0, blue: 0 },
    tripleThreat: { active: false, player: null, turnsLeft: 0 } };
}

function playout(game, state, random, cap) {
  for (let decision = 0; decision < cap; decision++) {
    const outcome = game.outcome(state);
    if (outcome) return outcome;
    const seat = game.actor(state);
    const actions = game.legalActions(state, seat);
    if (!actions.length) return null;
    state = game.apply(state, seat, actions[Math.floor(random() * actions.length)]);
  }
  return game.outcome(state);
}

/** ``count`` scenarios {group, scenario} from ``seed``: finish rate >= ``finish`` within ``cap`` decisions over
 * ``trials`` random playouts, each colour winning at least ``minWin`` of finished games. */
export async function chesstScenarios(rulesPath, { count = 6, seed = 7, cap = 40, trials = 20, finish = 0.6,
  minWin = 0.2, attempts = 4000 } = {}) {
  const game = await createChesstGame(rulesPath);
  const random = rng(seed), out = [];
  for (let attempt = 0; attempt < attempts && out.length < count; attempt++) {
    const candidate = position(random);
    let state;
    try { state = game.initialize({ state: candidate }, 1); } catch { continue; }
    if (game.outcome(state)) continue;
    const wins = { gold: 0, blue: 0 }; let finished = 0;
    const playRandom = rng(seed * 1000003 + attempt);
    for (let t = 0; t < trials; t++) {
      const result = playout(game, state, playRandom, cap);
      if (!result) continue;
      finished++;
      for (const seat of ['gold', 'blue']) if (result.scores[seat] === 1) wins[seat]++;
    }
    if (finished / trials < finish || wins.gold < minWin * finished || wins.blue < minWin * finished) continue;
    out.push({ group: `chesst/generated-s${seed}-a${attempt}`, family: 'chesst', scenario: { state },
      selection: { attempt, trials, finished, wins, cap } });
  }
  if (out.length < count) throw Error(`only ${out.length} ChessT scenarios met the filter`);
  return { game, scenarios: out };
}
