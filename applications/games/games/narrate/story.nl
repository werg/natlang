---
description: Tell what happened in one turn of a game world, in a few plain sentences, from its events.
args:
  kind: "'economy' | 'combat' | 'npc'"
  events: GameEvent[]
returns: string
---
Tell the turn of a world of kind kind from events, the facts the world recorded, in order. Use only what the events
hold.

1. economy: the event "economy.settle" has outcomes. Say for each traded outcome who bought what from whom and for how
   much; say for each rejected outcome who could not trade and why (its reason); say which merchants passed.
2. combat: the event "combat.resolve" has the round, the fighters afterwards (x, hp, cooldown) and the damage each
   fighter took. Say who was hurt and by how much, who is down (hp 0), and where the fighters stand.
3. npc: the event "npc.act" has actor, said, action and the inventory and commitments afterwards. Say what the NPC
   answered and what it did.

Return two to four sentences in the present tense, without lists.
