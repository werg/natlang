# Typed text reader: one system header

The supervisor found repeated successful-parent teacher traces whose sampled
reader actions failed before inference with `System message must be at the
beginning.` The shared text-emulation adapter prepends a new system message to
an ordinary reader request that already starts with one. A strict model chat
template rejects the second system message. This is a transport bug, not evidence
that the teacher cannot perform the read.

The existing typed read is the natlang program. Its declared library instruction,
input value and result type continue to define the operation. This change adds
no decision policy or alternate reader.

1. Render the authenticated existing request and input block bodies as before.
2. Put the existing readout instruction before the existing first system
   message's content. Retain that message's other fields. For content parts,
   prepend a text part and preserve every original part in order.
3. If there is no initial system message, add the instruction in one system
   message before the existing request. Preserve the remaining messages,
   tools, typed result validation and signal.
4. Record prompt revision11 and actual rendered request provenance. Retain all
   historical revision10 artifacts without rewriting their source identities.
5. Prove the real typed library read succeeds through a provider that rejects
   any noninitial system message. Cover both string and part-array headers and
   confirm the input request remains unchanged.
6. Before adopting model-facing request assembly, run at least48 live typed
   reads through our self-hosted teacher, including string/value transport and
   nested reads. Stage a new runtime for future batches; preserve existing
   batch processes and authenticate the guardian/exporter transition together.

Header assembly and provenance are crisp protocol mechanisms. The typed read
itself remains the existing natural-language library function. This fix is
shared by every text-emulation teacher/model transport. Native shared-state
training and inference retain their own authentic state protocol; this text
transport grants no communication-foundation qualification.
