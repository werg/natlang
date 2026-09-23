export default function attach(state: NotebookState, answer: string): NotebookState {
return { ...state, answer: answer };
}
