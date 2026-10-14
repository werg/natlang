# SearchSwarm/OWL CPU preparation on the existing LFM sandbox

Policy is this natural-language procedure: use two spare host CPUs at nice19
only while real LFM host headroom permits it; preserve training/export priority;
fetch exact public pins, convert with shared adapters, protected-finalize, then
replay existing native call programs. No GPU model job or extra sandbox. Copies,
conversion and successful replay do not replace protected admission/registration.

## Decomposition

1. Check resources and dependencies with crisp external-service probes; choose
   two actually permitted CPUs from the reported affinity set.
2. Stage a frozen Git archive and exact protected/reservation data files.
   Git is the source transport; source trees are not copied between machines.
3. Fetch the two declared public dataset revisions and verify full SHA/size;
   this is existing acquisition plumbing and records license provenance.
4. Invoke existing `convert-agents` adapters and streaming `finalize`, retaining
   split groups, upstream sessions, rejected rows and all protected hits.
5. Invoke the existing shared `port_record_calls.sh` pipeline with one builder
   and one replayer in the isolated snapshot. Its repository root and builder
   count become explicit environment settings; canonical defaults stay intact.
   This is external execution plumbing, not a second native converter.
6. Preserve terminal receipts, manifests, omissions and compressed native turns;
   report to the supervisor for exact checks and later rendering/admission.

## Measured preparation

LFM sandbox7ee701ae-8e56-49d8-b97a-594f7eb33fe9 at22:28UTC had
16,426,283,008hostbytes used out of64GiB, zero memory events/OOM and29.86TB free
root storage. Allowed CPU affinity0..8 supports choosing7,8. Public source pins:
SearchSwarm/SearchSwarm-SFT4bad3397be6144a0fbbaac3121929e082f32865e and
camel-ai/OWL-SFT878b41ca2eb68d2247145f303a854adedd3f2fb3,
2,128,554,844bytes total. HuggingFaceHub, PyArrow, jsonschema and Torch exist;
Node24 and zstd CLI exist. Python zstandard is absent; these stages use the
existing zstd CLI and do not require that module. Protected index is missing on
LFM and must be staged with SHA
585b21e7091adfbc4f851ef35b0b3c6da00f032fffe69a8b98816a567abad7bf.
The existing Node dependencies are present; compile the isolated Git snapshot's
own TypeScript dist without changing the active trainer source/runtime.
