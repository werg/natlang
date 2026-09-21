/*---
engine: typescript-host
args:
  state: State
returns: View
---*/
return {
  heading: args.state.question || 'Ask something worth investigating.',
  summary: args.state.notice,
  active_view: args.state.active_view,
  suggestions: [],
};
