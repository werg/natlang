# Retained collector responses keep their native artifact closure

Mellum's retained RAW round replayed exact saved model responses into a new
collector. The remote actor store and the collector's fresh memory store lacked
three original native states. Restoring the remote store alone does not repair
the collector: the journal replay branch bypasses the live transport's archival.
The agent correctly rejects a producer write whose artifact is unavailable.

1. Retain the saved response, request hash, sampling receipts and exact state IDs.
2. Use one shared reply-archival operation for live transport and journal replay.
   Extract written states, native output references and sampling input/raw-write
   roots; recursively archive their existing typed dependency DAG through
   `archiveNativeArtifactFrom`. Preserve the existing ordinary block path too.
3. Hydrate the collector store before returning the recorded response to the
   agent. Missing or changed remote artifacts abort the whole collector through
   its existing fatal-error mechanism: a child may not recover by sampling a
   replacement turn and earning a false successful row. Do not regenerate
   state, erase references, resample a saved response or fabricate completion.
4. Exercise an interrupted real collector run resumed into a fresh memory store,
   with an exact native producer/consumer graph and no repeat model request for
   the saved prefix. Check artifact custody and unchanged sampling provenance;
   also reject an unavailable state rather than silently substituting a turn.

This is crisp content-addressed transport and durability mechanism. Task policy
and model-facing prompts do not change. Sandbox recovery must restore the exact
remote artifact closure before resuming; the client fix cannot invent missing
bytes. Publish a fresh runtime for failed cases while healthy sealed consumers
continue with their existing runtime.
