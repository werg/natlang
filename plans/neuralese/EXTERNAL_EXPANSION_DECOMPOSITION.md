# External training expansion

Owner request, 2026-10-14: integrate the sources in
`Natlang_Neuralese_Training_Data_Expansion_Plan.md`. Static conversion is sufficient;
retain useful general coding and tool competence alongside Neuralese communication.
All licenses are accepted and recorded as provenance. Access failures are acquisition
facts, not license holds. Existing training and the readout repair continue.

## Natural-language stages

1. `resolveSources(plan) -> SourceCatalog`: enumerate every named source, pin its
   current immutable revision and files, and record declared license and access.
   Prefer communication sources and already supported trajectory formats first.
2. `acquireSource(source) -> SourceSnapshot`: acquire available content, starting
   with complete manageable releases and breadth across larger releases. Record
   partial acquisition and outstanding files explicitly. Hash every acquired file.
3. `normalizeSource(snapshot) -> SessionGraph`: preserve original sessions, roles,
   tools, observations and source identities. Join observed dispatch/return/peer
   edges only when their original identifiers or exact payloads establish them.
   Keep unknown outcomes and partial graphs available for appropriate objectives.
4. `lowerTraining(graph) -> TrainingViews`: reuse the existing trajectory adapter
   and typed history handover representation. Preserve observed communication;
   label statically created history/provider boundaries as derived communication.
   Retain coding/tool conversations that need further native lowering. Never call
   reconstructed external evidence a validated native runtime edge.
5. `publishTraining(views) -> CorpusBindings`: inherit upstream held splits and
   project reservations; group all sessions of a task together. Register immutable
   manifests, render both model lines, bind actual recipe inputs, and report actual
   exposure. Copies and schema checks do not alone establish training admission.

## Crisp mechanisms

Hub APIs/CLI, JSON/Parquet decoders, hashes, disk indexes, exact joins, durable
publication, split verifiers and existing token renderers are crisp because they
must reproduce bytes and interfaces. Source routing and objective eligibility are
the natural-language policy above, expressed as explicit catalog/admission facts.
Extend `scripts/neuralese_data/agents.py`, `trajectory.py`, shared write-site
extraction and the E2E miner; do not introduce another trainer or channel format.

External data keeps original tool semantics unless a static mapping has an exact
native service equivalent. New instruction wording is model-facing measurement
debt until tested live; data conversion is not an executor prompt replacement.

## First wave

Resolve all forty catalog entries. Acquire WideSeek, SearchSwarm and CooperBench
communication releases, then worker/research/tool sources and supported coding
releases. Deduplicate CooperBench aggregates and already acquired SWE lineage.
Convert what is usable now rather than waiting for complete native recollection.
Record the source-neutral graph and every omission or outstanding conversion.

## Completion evidence

Pinned catalog and acquired SHA manifests; converted unique sessions/targets and
communication edges with their provenance; protected splits; both tokenizer views;
actual shared recipe bindings and sampler exposure; consumer-to-producer gradients
for E2E views; matched held results. Acquisition or a small pilot alone is not the
completed expansion.

## Direct compressed union output

The MuSiQue family hold requires a corrected broad union before the next main-line launch. Keep the existing admission, protected split, deduplication and tokenizer-twin decisions; explicitly remove the held member. Serialize selected rows directly into zstd, recording compressed identity plus independently decoded SHA-256 and byte count. Compression and durability extend the existing crisp union builder and compressed registry contract. This avoids another roughly140GB raw working copy without changing model-visible data or granting source admission.
