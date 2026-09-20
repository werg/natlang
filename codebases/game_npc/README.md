# General NPC interactions

Natlang reads an assigned event and that NPC's memory, inventory and
commitments, then writes dialogue and one exact action. The host issues an
observation identity, rejects forged or replayed observations, and preserves
inventory and promise provenance. This extends the shopkeeper pattern to
event-sourced NPC memory without exposing other actors' private state.
