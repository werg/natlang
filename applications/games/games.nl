---
description: Play one turn of a game world (a market of merchants, a duel arena, or a village of NPCs) from a scene. The world decides on its snapshot and commits once; the report carries the next state, the events and the narration.
args:
  scene: Scene
  settings: Settings
returns: TurnReport
---
Play one turn of scene with the stages in your folder. Every world follows the same turn: observe a snapshot, decide,
validate, resolve, then commit once. scene.state is the snapshot; a turn never changes it, and the next state is in
the turn's result.

1. By scene.kind:
   - "economy": turn = economy(scene.state, settings).
   - "combat": turn = combat(scene.state, settings).
   - "npc": turn = npc(scene.state, scene.actor, scene.event, settings).
2. narration = narrate(scene.kind, turn.events, settings). When turn.ok is false the events are empty and narration is
   the sentence "The turn did not happen: " followed by turn.problem.
3. Return the turn's ok, state, events, log and problem, with kind as scene.kind and narration.
