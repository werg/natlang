---
args:
  perception: CombatObservation
returns: CombatPlan
---
Choose left, stay or right and attack, guard or rest. Attack names a visible
opponent. Movement and attacks resolve together after all fighters submit.
Consider health, distance and cooldown from the supplied observation. The
simulation, not this plan, decides collision, damage and death.
