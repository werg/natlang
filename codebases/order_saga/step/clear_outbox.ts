export default function clear_outbox(state: State): State {
return {...state, outbox: []};
}
