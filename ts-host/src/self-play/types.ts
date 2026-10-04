/** Authority stays with the host game; policies receive only their own observation. */
export type GameOutcome = { scores: Record<string, number>; reason: string };
/** An illegal submitted action is a gameplay loss; other engine errors are infrastructure failures. */
export class IllegalGameAction extends Error {
  constructor(message: string) { super(message); this.name = 'IllegalGameAction'; }
}
export type GameSpec = {
  id: string;
  revision: string;
  rules: string;
  seats: string[];
  initialize(scenario: unknown, seed: number): unknown;
  actor(state: unknown): string;
  observe(state: unknown, seat: string): unknown;
  legalActions?(state: unknown, seat: string): unknown[];
  apply(state: unknown, seat: string, action: unknown): unknown;
  outcome(state: unknown): GameOutcome | null;
};
export type GameDecision = { game: string; rules: string; seat: string; decision: number;
  observation: unknown; legalActions?: unknown[] };
export type GamePolicy = { id: string; decide(view: GameDecision, signal?: AbortSignal): Promise<unknown> };
export type GameFrame = { index: number; seat: string; policy: string; view: GameDecision;
  action: unknown; before: string; after: string };
export type GameMatch = { schema: 'natlang.adversarial-match/1'; game: string; revision: string;
  seed: number; scenario: unknown; initial: unknown; frames: GameFrame[];
  policies: Record<string, string>; maxDecisions: number;
  disposition: 'completed' | 'illegal-action' | 'incomplete'; outcome: GameOutcome | null;
  error?: string; rejected?: { seat: string; view: GameDecision; action: unknown; reason: string };
  final: unknown; identity: string };
