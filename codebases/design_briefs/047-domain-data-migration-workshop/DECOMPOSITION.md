# Domain data migration workshop

Source meaning is recovered from dictionaries, domain-owner explanations, application context, and representative records. Reused identifiers carry their system and validity interval. Null, unknown, not applicable, and withheld values remain distinct unless an owner explicitly accepts a declared information loss.

Mappings state their conditions and target representation before bounded record resolution begins. Records are accepted only when they satisfy both target schema and source semantics; unrepresentable but valid cases are quarantined rather than fabricated. Rehearsal compares exact counts, relationships, invariants, and round trips, then captures changes after extraction under the frozen mapping version. Cutover stays blocked until every delta and source record is accounted for, and rollback preserves target-created work through an explicit reconciliation plan.
