# Semantic media workbench

`transform.nl` runs a full batch transformation: probe the source, choose one
recipe semantically, render it through FFmpeg, inspect the result, assess the
user's intent, and finalize only when the exact metadata checks agree. It
supports trim, crop, scale, and transcode plans. Unsupported operations receive
an explicit unsupported receipt. Natlang chooses and assesses; crisp functions
enforce source/output identity and technical postconditions.

The native `MediaWorkspace` keeps video bytes, temporary sample frames, and
FFmpeg processes outside the natlang value tree. It admits only paths inside a
workspace, refuses preexisting outputs, invokes FFmpeg without a shell, probes
streams and duration with ffprobe, hashes the source and output, and labels
interrupted renders as unknown because an output may already exist. An optional
`vision` callback receives a sampled frame and returns a verdict plus model ID.
Without that callback, a technically valid crop remains in `review` rather
than claiming visual success. The trace records probes, renders, and the visual
inspector's identity; native media state is not replayable from the trace.

`ts-host/test/media-workbench.test.mjs` runs real CPU FFmpeg fixtures for trim,
crop, visual review, sampled-frame inspection, invalid geometry, and corrupt
input. It supplies a scripted model callback, so these tests establish the
host/program contract and exact checks, not live-model semantic quality.

The next product gates are a teacher pilot on varied natural requests,
independent review of visual judgments, bounded revision after a failed
inspection, and an interactive Fold over request/job/cancellation events. A
long native render must report its actual outcome after cancellation; a
timeout cannot be interpreted as rollback. These facilities stay in the media
host unless a second application demonstrates a genuinely shared contract.
