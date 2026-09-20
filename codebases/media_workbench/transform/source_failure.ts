/*---
engine: typescript-host
args:
  request: Request
  source: Clip
returns: MediaResult
---*/
const receipt = { status: 'failed', output: args.request.output, exit_code: -1, sha256: '', detail: args.source.detail };
const inspection = { status: 'unavailable', width: 0, height: 0, duration: 0, has_audio: false,
  sha256: '', visual_status: 'unavailable', visual_detail: '', detail: 'source unavailable' };
return { status: 'failed', output: args.request.output, technical_ok: false, semantic_ok: false,
  needs_visual_review: false, explanation: args.source.detail, receipt, inspection };
