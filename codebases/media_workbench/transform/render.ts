export default async function render(request: Request, source: Clip, plan: Plan): Promise<Receipt> {
if (plan.input !== request.input || plan.output !== request.output ||
    source.id !== request.input || source.status !== 'ok')
  return { status: 'failed', output: request.output, exit_code: -1, sha256: '',
    detail: 'plan changed the selected input or output' };
return await host.media.render(plan);
}
