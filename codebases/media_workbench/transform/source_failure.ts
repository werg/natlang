export default function source_failure(request: Request, source: Clip): MediaResult {
const receipt = { status: 'failed', output: request.output, exit_code: -1, sha256: '', detail: source.detail };
const inspection = { status: 'unavailable', width: 0, height: 0, duration: 0, has_audio: false,
  sha256: '', visual_status: 'unavailable', visual_detail: '', detail: 'source unavailable' };
return { status: 'failed', output: request.output, technical_ok: false, semantic_ok: false,
  needs_visual_review: false, explanation: source.detail, receipt, inspection };
}
