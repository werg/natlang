export default function finished(state: State): boolean {
return state.status !== 'running';
}
