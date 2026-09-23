export default function complete(state: NotebookState): boolean {
return state.status !== 'running';
}
