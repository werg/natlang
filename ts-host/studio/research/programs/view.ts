import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "./types.js";

export default function view(state: State): View {
return {
  heading: state.question || 'Ask something worth investigating.',
  summary: state.notice,
  active_view: state.active_view,
  suggestions: [],
};
}
