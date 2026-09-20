/*---
engine: typescript-host
args:
  request: Request
  source: Clip
  plan: Plan
returns: Receipt
---*/
if (args.plan.input !== args.request.input || args.plan.output !== args.request.output ||
    args.source.id !== args.request.input || args.source.status !== 'ok')
  return { status: 'failed', output: args.request.output, exit_code: -1, sha256: '',
    detail: 'plan changed the selected input or output' };
return await host.media.render(args.plan);
