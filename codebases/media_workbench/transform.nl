---
description: Interpret a media request, run one exact transform, inspect the output, and report uncertainty honestly.
args:
  request: Request
returns: MediaResult
---
function transform(request) -> MediaResult
  source = probe(request.input)
  if source.status is not "ok":
    return source_failure(request, source)
  plan = choose(request, source)
  receipt = render(request, source, plan)
  inspection = inspect(request, plan, receipt)
  assessment = assess(request, source, plan, receipt, inspection)
  return finalize(request, source, plan, receipt, inspection, assessment)
