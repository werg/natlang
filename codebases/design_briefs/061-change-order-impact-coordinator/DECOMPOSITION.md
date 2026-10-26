# Change-order impact design

The coordinator separates three records that are often conflated: who instructed a change, what work actually occurred, and when commercial authority was confirmed. It limits a drawing update to changed areas, traverses only explicit dependencies, tests schedule effect against supplied activity float, and deduplicates cost attribution across overlapping changes. The output preserves disputed directions and pre-approval events without backdating approval or claiming a delay that the supplied float does not support.
