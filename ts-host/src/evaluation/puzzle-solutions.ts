// Exact verifiers for puzzles with many valid solutions (oracle normalization `puzzle-solution`). The oracle's
// `expected` is the puzzle instance, not one solution: an answer is accepted when it solves that instance.
//   countdown: { numbers, target, use_all? } and an arithmetic expression over + - * / and parentheses whose
//              integer literals are drawn from `numbers` (each at most once; every one when `use_all`), evaluated
//              exactly with rationals (no rounding, division by zero fails).
//   n-queens:  { n, fixed: [row, col][] } and the queens as {row, col} records or [row, col] pairs (0-based); the
//              queens together with `fixed` must be n queens on the board, no two sharing a row, column or diagonal.
//   sokoban:   { grid: string[] } in XSB notation (# wall, @ player, + player on goal, $ box, * box on goal, . goal,
//              space floor) and a move string over L R U D (case-insensitive, spaces ignored); after the moves every
//              box is on a goal. A move into a wall or pushing a box into a wall or box fails.

export const PUZZLE_SOLUTION_NORMALIZATIONS = ['puzzle-solution'] as const;
export type PuzzleKind = 'countdown' | 'n-queens' | 'sokoban';

type Rational = { n: bigint; d: bigint };
const gcd = (a: bigint, b: bigint): bigint => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
function rational(n: bigint, d: bigint): Rational {
  if (d === 0n) throw new RangeError('division by zero');
  if (d < 0n) { n = -n; d = -d; }
  const g = gcd(n, d) || 1n;
  return { n: n / g, d: d / g };
}

/** Exact value of an arithmetic expression and the integer literals it uses, or undefined when it does not parse. */
export function evaluateArithmetic(text: string): { value: Rational; literals: bigint[] } | undefined {
  const tokens = text.replace(/×/g, '*').replace(/÷/g, '/').match(/\d+|[-+*/()]|\S/g) ?? [];
  let at = 0;
  const literals: bigint[] = [];
  const peek = () => tokens[at];
  function primary(): Rational {
    const token = tokens[at++];
    if (token === '(') { const value = sum(); if (tokens[at++] !== ')') throw new SyntaxError('expected )'); return value; }
    if (token !== undefined && /^\d+$/.test(token)) { const n = BigInt(token); literals.push(n); return { n, d: 1n }; }
    throw new SyntaxError(`unexpected ${token ?? 'end'}`);
  }
  function product(): Rational {
    let value = primary();
    while (peek() === '*' || peek() === '/') {
      const op = tokens[at++], right = primary();
      value = op === '*' ? rational(value.n * right.n, value.d * right.d) : rational(value.n * right.d, value.d * right.n);
    }
    return value;
  }
  function sum(): Rational {
    let value = product();
    while (peek() === '+' || peek() === '-') {
      const op = tokens[at++], right = product();
      value = rational(op === '+' ? value.n * right.d + right.n * value.d : value.n * right.d - right.n * value.d, value.d * right.d);
    }
    return value;
  }
  try {
    const value = sum();
    return at === tokens.length && tokens.length ? { value, literals } : undefined;
  } catch { return undefined; }
}

function countdownSolves(actual: unknown, instance: Record<string, unknown>): boolean {
  const numbers = instance.numbers, target = instance.target;
  if (typeof actual !== 'string' || !Array.isArray(numbers) || typeof target !== 'number' || !Number.isInteger(target)) return false;
  const parsed = evaluateArithmetic(actual.replace(/=\s*-?\d+\s*$/, ''));
  if (!parsed) return false;
  const pool = new Map<string, number>();
  for (const n of numbers) pool.set(String(n), (pool.get(String(n)) ?? 0) + 1);
  for (const literal of parsed.literals) {
    const left = pool.get(String(literal)) ?? 0;
    if (!left) return false;
    pool.set(String(literal), left - 1);
  }
  if (instance.use_all === true && [...pool.values()].some(left => left > 0)) return false;
  return parsed.value.d === 1n && parsed.value.n === BigInt(target);
}

function queensSolve(actual: unknown, instance: Record<string, unknown>): boolean {
  const n = instance.n;
  if (!Array.isArray(actual) || typeof n !== 'number' || !Number.isInteger(n) || n < 1) return false;
  const cell = (item: unknown): [number, number] | undefined => {
    const pair = Array.isArray(item) ? item : item && typeof item === 'object' ?
      [(item as Record<string, unknown>).row, (item as Record<string, unknown>).col] : undefined;
    if (!pair || pair.length !== 2 || !pair.every(v => Number.isInteger(v) && (v as number) >= 0 && (v as number) < n)) return undefined;
    return pair as [number, number];
  };
  const placed = new Map<string, [number, number]>();
  for (const item of [...(Array.isArray(instance.fixed) ? instance.fixed : []), ...actual]) {
    const c = cell(item);
    if (!c) return false;
    placed.set(c.join(','), c);
  }
  const queens = [...placed.values()];
  if (queens.length !== n) return false;
  const distinct = (key: (q: [number, number]) => number) => new Set(queens.map(key)).size === n;
  return distinct(q => q[0]) && distinct(q => q[1]) && distinct(q => q[0] - q[1]) && distinct(q => q[0] + q[1]);
}

function sokobanSolves(actual: unknown, instance: Record<string, unknown>): boolean {
  const grid = instance.grid;
  if (typeof actual !== 'string' || !Array.isArray(grid) || !grid.every(row => typeof row === 'string')) return false;
  const walls = new Set<string>(), goals = new Set<string>(), boxes = new Set<string>();
  let player: [number, number] | undefined;
  (grid as string[]).forEach((row, r) => [...row].forEach((ch, c) => {
    const key = `${r},${c}`;
    if (ch === '#') walls.add(key);
    if ('.+*'.includes(ch)) goals.add(key);
    if ('$*'.includes(ch)) boxes.add(key);
    if ('@+'.includes(ch)) player = [r, c];
  }));
  if (!player || !boxes.size || boxes.size !== goals.size) return false;
  const moves = actual.replace(/[\s,]/g, '').toUpperCase();
  if (!/^[LRUD]*$/.test(moves)) return false;
  const step: Record<string, [number, number]> = { L: [0, -1], R: [0, 1], U: [-1, 0], D: [1, 0] };
  for (const move of moves) {
    const [dr, dc] = step[move]!;
    const next = `${player[0] + dr},${player[1] + dc}`;
    if (walls.has(next)) return false;
    if (boxes.has(next)) {
      const beyond = `${player[0] + 2 * dr},${player[1] + 2 * dc}`;
      if (walls.has(beyond) || boxes.has(beyond)) return false;
      boxes.delete(next); boxes.add(beyond);
    }
    player = [player[0] + dr, player[1] + dc];
  }
  return [...boxes].every(box => goals.has(box));
}

export function puzzleSolves(actual: unknown, instance: unknown, puzzle: unknown): boolean {
  if (!instance || typeof instance !== 'object' || Array.isArray(instance)) return false;
  const spec = instance as Record<string, unknown>;
  if (puzzle === 'countdown') return countdownSolves(actual, spec);
  if (puzzle === 'n-queens') return queensSolve(actual, spec);
  if (puzzle === 'sokoban') return sokobanSolves(actual, spec);
  throw new RangeError(`unknown puzzle-solution puzzle: ${String(puzzle)}`);
}
