import { describe, expect, it } from 'vitest';
import { getGEPAUpdateGroup } from './gepaDependencies.js';

describe('getGEPAUpdateGroup', () => {
  it('returns a dependency bundle rooted at the selected component', () => {
    const targets = [
      { id: 'root::instruction', kind: 'instruction', current: 'a' },
      {
        id: 'root::description',
        kind: 'description',
        current: 'b',
        dependsOn: ['root::instruction'],
      },
    ];

    expect(getGEPAUpdateGroup(targets[1]!, targets).map((t) => t.id)).toEqual([
      'root::description',
      'root::instruction',
    ]);
  });
});
