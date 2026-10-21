# Teacher model handover

The owner asks the supervisor to evaluate RecursiveSI/flint on actual natlang
tasks and replace the self-hosted Qwen teacher if useful. That choice remains a
natural-language supervisory decision based on captured task outcomes and actual
throughput. A model card or a successful server launch is insufficient.

1. Select source-bound training tasks and capture complete executions using the
   existing collector. Keep protected evaluation separate.
2. Compare completed task outcomes with the existing teacher, distinguishing
   infrastructure failures from model behavior.
3. If selected, bind the new teacher's repository/revision, quantized artifact,
   sampling settings and live serving identity in an immutable source descriptor.
4. Pass the same descriptor and endpoint through primary and overlapping feeders.
   Existing batches retain their original actor and provenance.
5. Admit independently verified usable trajectories through the existing shared
   conversion and append mechanisms for both students. Registration and collection
   do not establish learner exposure.

Exact byte bindings, process identity checks, environment transport and collector
arguments are crisp helpers in the existing guardian and feeder. They enforce the
supervisor's declared selection; they do not choose a teacher or invent quality
credit. There is one generation pipeline for all self-hosted teachers.

Feeder custody includes the short normal `nice -n 10 python -u <exact feeder>`
exec transition immediately after spawning. Authenticate this exact argument
vector and process birth, then require the ordinary sealed source/frontier checks
before granting refill capacity. A launch race must not terminate supervision of
an already collecting batch; unrelated wrappers remain rejected.
