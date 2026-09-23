export default function recover(acc: Session, item: Event): Session {
const state = acc;
if (state.status !== 'running' && state.status !== 'cancel-requested') return state;
return { ...state, revision: state.revision + 1, active_request: '', active_job: '', status: 'unknown',
  messages: [...state.messages,
    `The prior host stopped while ${state.active_job} was active. Its outcome is unknown; inspect external effects before retrying.`] };
}
