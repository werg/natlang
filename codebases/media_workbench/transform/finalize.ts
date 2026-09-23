export default function finalize(request: Request, source: Clip, plan: Plan, receipt: Receipt, inspection: Inspection, assessment: Assessment): MediaResult {
const p = plan, s = source, r = receipt, i = inspection, a = assessment;
const expectedWidth = p.kind === 'crop' || p.kind === 'scale' ? p.width : s.width;
const expectedHeight = p.kind === 'crop' || p.kind === 'scale' ? p.height : s.height;
const expectedDuration = p.kind === 'trim' ? p.end - p.start : s.duration;
const technical_ok = r.status === 'ok' && i.status === 'ok' && r.output === request.output &&
  i.sha256 === r.sha256 && i.width === expectedWidth && i.height === expectedHeight &&
  Math.abs(i.duration - expectedDuration) <= 0.16 && i.has_audio === (p.keep_audio && s.has_audio);
const needs_visual_review = p.kind === 'crop' && i.visual_status !== 'supported' || a.needs_visual_review;
const semantic_ok = technical_ok && a.intent_met && i.visual_status !== 'contradicted';
const status = r.status === 'unknown' ? 'unknown' : r.status === 'unsupported' ? 'unsupported' :
  !technical_ok ? 'failed' : !semantic_ok ? 'rejected' : needs_visual_review ? 'review' : 'verified';
return { status, output: request.output, technical_ok, semantic_ok,
  needs_visual_review, explanation: technical_ok ? a.explanation : `Technical check failed: ${r.detail || i.detail}`,
  receipt: r, inspection: i };
}
