import type { AxGEPAComponentTarget } from './gepaComponents.js';

export type AxGEPAComponentBanditState = {
  proposals: number;
  accepts: number;
  lastAcceptIter: number;
  stagnation: number;
};

export class AxGEPAComponentSelector {
  private readonly states = new Map<string, AxGEPAComponentBanditState>();

  constructor(
    private readonly targets: readonly AxGEPAComponentTarget[],
    initialState?: Readonly<Record<string, AxGEPAComponentBanditState>>
  ) {
    for (const target of targets) {
      const existing = initialState?.[target.id];
      this.states.set(target.id, {
        proposals: Math.max(0, Math.floor(existing?.proposals ?? 0)),
        accepts: Math.max(0, Math.floor(existing?.accepts ?? 0)),
        lastAcceptIter: Math.floor(existing?.lastAcceptIter ?? -1),
        stagnation: Math.max(0, Math.floor(existing?.stagnation ?? 0)),
      });
    }
  }

  public getState(id: string): AxGEPAComponentBanditState | undefined {
    const state = this.states.get(id);
    return state ? { ...state } : undefined;
  }

  public snapshot(): Record<string, AxGEPAComponentBanditState> {
    return Object.fromEntries(
      [...this.states.entries()].map(([id, state]) => [id, { ...state }])
    );
  }

  public pick(iteration: number, rand: () => number): AxGEPAComponentTarget {
    if (this.targets.length === 1) return this.targets[0]!;
    if (rand() < 0.1) {
      return this.targets[Math.floor(rand() * this.targets.length)]!;
    }

    const totalProposals = Math.max(
      1,
      [...this.states.values()].reduce((sum, state) => sum + state.proposals, 0)
    );

    const weights = this.targets.map((target) => {
      const state = this.states.get(target.id)!;
      const acceptRate =
        state.proposals === 0 ? 0 : state.accepts / state.proposals;
      const proposalPressure = state.proposals / totalProposals;
      const staleBoost =
        state.lastAcceptIter < 0
          ? Math.min(iteration + 1, 10)
          : Math.min(iteration - state.lastAcceptIter, 10);
      return (
        1.4 * (1 - acceptRate) +
        0.8 * state.stagnation +
        0.2 * staleBoost -
        0.7 * proposalPressure
      );
    });

    const max = Math.max(...weights);
    const exp = weights.map((weight) => Math.exp(weight - max));
    const total = exp.reduce((sum, weight) => sum + weight, 0);
    let threshold = rand() * total;
    for (let i = 0; i < exp.length; i++) {
      threshold -= exp[i]!;
      if (threshold <= 0) return this.targets[i]!;
    }
    return this.targets[this.targets.length - 1]!;
  }

  public recordProposal(id: string): void {
    const state = this.states.get(id);
    if (!state) return;
    state.proposals += 1;
  }

  public recordResult(id: string, accepted: boolean, iteration: number): void {
    const state = this.states.get(id);
    if (!state) return;
    if (accepted) {
      state.accepts += 1;
      state.lastAcceptIter = iteration;
      state.stagnation = 0;
    } else {
      state.stagnation += 1;
    }
  }
}
