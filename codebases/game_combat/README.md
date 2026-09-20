# Actor combat game

Natlang chooses movement and a tactic once per round. The host resolves all
submitted movement before simultaneous legal attacks, cooldown and damage.
Stale plans are refused. The headless arena keeps physics and health exact;
rendering and lower-frequency tactics in a live frame loop are later host work.
