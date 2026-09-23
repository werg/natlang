---
description: Interpret a media request, run one exact transform, inspect the output, and report uncertainty honestly. Use files to inspect input and sidecar metadata when needed.
args:
  request: Request
  files?: Record<string, File>
returns: MediaResult
---
function transform(request, files) -> MediaResult
  source = probe(request.input)
  if source.status is not "ok":
    return source_failure(request, source)
  plan = choose(request, source, files)
  receipt = render(request, source, plan)
  inspection = inspect(request, plan, receipt)
  assessment = assess(request, source, plan, receipt, inspection, files)
  return finalize(request, source, plan, receipt, inspection, assessment)
