---
description: Recover only source-supported event precedence within each frame.
args:
  frames: FrameIndex
  entities: EntityIndex
  canon: SourceRecord[]
returns: TemporalGraph
---
1. Inventory explicit before, after and simultaneous relations in events and canon. Resolve names
   using entities without replacing frame ownership. Attach exact quotations to Precedence edges.
2. Interpret clock and bell references under their stated event definitions. The port bell and tide
   bell remain different anchors unless canon explicitly equates them.
3. Add implicit precedence only when a described action entails it: reading follows obtaining access;
   an explicit later transfer follows the holder's acquisition. Record that semantic entailment's witness.
4. Preserve ambiguous temporal pairs. A recollection recounted later does not place its remembered
   events after the selected scene; narration order and narrated event order are distinct relations.
5. Close reachable precedence paths within a frame. Each pass adds only previously absent ordered
   pairs from the finite event inventory and stops when no pair is added.
6. If an event reaches itself, record the involved IDs in cycles. A cycle does not authorize rewriting
   the chronology to the visually simplest order. Later knowledge checks must retain the conflict.
7. Return direct supported edges plus unresolvedPairs and cycles. Do not emit a total sorted event
   list that fills gaps the sources leave open.
8. This child owns chronology. The caller consumes its graph; physical and epistemic branches may
   test reachability but cannot add a new timeline premise to make their own findings easier.
