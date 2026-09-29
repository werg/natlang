import { describe, expect, it } from 'vitest';
import { AxGEPAComponentSelector } from './gepaSelection.js';

describe('AxGEPAComponentSelector', () => {
  it('prefers stagnated components over recently accepted ones', () => {
    const selector = new AxGEPAComponentSelector([
      { id: 'recent::instruction', kind: 'instruction', current: 'a' },
      { id: 'stale::instruction', kind: 'instruction', current: 'b' },
    ]);

    selector.recordProposal('recent::instruction');
    selector.recordResult('recent::instruction', true, 5);
    for (let i = 0; i < 4; i++) {
      selector.recordProposal('stale::instruction');
      selector.recordResult('stale::instruction', false, i);
    }

    let recent = 0;
    let stale = 0;
    const rand = () => 0.5;
    for (let i = 0; i < 50; i++) {
      const picked = selector.pick(10, rand);
      if (picked.id === 'recent::instruction') recent++;
      if (picked.id === 'stale::instruction') stale++;
    }

    expect(stale).toBeGreaterThan(recent);
  });

  it('can persist and restore selector state', () => {
    const targets = [
      { id: 'a::instruction', kind: 'instruction', current: 'a' },
      { id: 'b::instruction', kind: 'instruction', current: 'b' },
    ];
    const selector = new AxGEPAComponentSelector(targets);
    selector.recordProposal('a::instruction');
    selector.recordResult('a::instruction', false, 1);

    const restored = new AxGEPAComponentSelector(targets, selector.snapshot());
    expect(restored.getState('a::instruction')).toMatchObject({
      proposals: 1,
      accepts: 0,
      stagnation: 1,
    });
  });
});
