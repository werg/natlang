export default function append(state: ConsoleState, request: string, result: NotebookState): ConsoleState {
return { requests: [...state.requests, request],
  runs: [...state.runs, result], status: result.status };
}
