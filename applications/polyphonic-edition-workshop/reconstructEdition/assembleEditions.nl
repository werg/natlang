---
description: Assemble globally consistent source editions from actual location alternatives.
args:
  readings: ReadingIndex
returns: EditionSet
---
1. Check readings.bundleOrder names every actual bundle event exactly once. Select the actual
   bundle objects in that order, preserving readings.bundleOrder as the immutable traversal record.
   Initialize one empty hypothesis with no choices or claim assignments and a HypothesisFrontier
   containing those ordered original bundles.
2. Call advanceHypothesis.iterateOn with measure pendingBundles.length. A step can increase hypothesis
   count while still consuming exactly one original bundle; termination does not depend on pruning.
3. Retain rejected extension witnesses and unresolved compatibility separately from accepted complete
   hypotheses. Missing local options cannot become an implicit unchanged reading.
4. Check every surviving hypothesis contains one actual option per original bundle and a consistent
   source-level target per exclusive editorial claim. Generated IDs carry no semantic preference.
5. Return actual hypotheses, rejected extensions and unresolved issues. Reprises will refer to these
   same chosen source readings; they do not sample editorial attachment again.
