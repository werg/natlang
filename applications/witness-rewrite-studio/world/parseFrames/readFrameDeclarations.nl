---
description: Interpret explicit discourse declarations without turning context facts into scene events.
args:
  records: SourceRecord[]
returns: FrameDeclarationIndex
---
1. Initialize frames, contextSourceIds and unresolved empty. Visit each original SourceRecord in
   records once, preserving its id, path, text and sha256 for every source-linked contribution.
2. Read the source's declared authority and discourse wording. Identify explicit frame identities,
   kinds, owners and parent references. A statement about the actual world establishes its scope;
   it does not establish a new narrated event or a scene merely by mentioning an event identity.
3. When a source declares a frame, construct its FrameDefinition from that declaration. Leave
   sceneIds empty unless the declaration explicitly assigns scenes. A selected-world canon can
   declare ACTUAL while remaining context for later authority evaluation rather than an EventView.
4. For every declaration, choose exact quotation bytes from this record. Bind Witness.sourceId
   from record.id and Witness.path from record.path. Check those witnesses immediately with
   storyEvidence.checkWitnesses({revision: record.sha256, records: [record]}, witnesses).
5. Retain only declaration witnesses whose exact checks pass. If the source does not decide a
   kind, owner or parent, record the missing premise in unresolved; preserve explicit null owners
   for narrator-established selected frames rather than inventing a speaking character.
6. Merge compatible repeated declarations of the same frame identity. Union explicitly assigned
   scene IDs and exact witnesses. If kinds, owners or parent relations conflict, retain that conflict
   in unresolved and do not use a convenient declaration to erase the competing source.
7. Put each visited record.id in contextSourceIds. This is the inventory of context evidence that
   contributed declarations or authority wording, not a list of scene events. An instruction that
   quotes a proposed change remains an instruction and does not add that proposed action to frames.
8. Return frames, contextSourceIds and unresolved. The caller consumes declared parent availability;
   evaluateAuthority separately interprets controlling canon propositions using their source role.
