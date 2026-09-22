export type Task = { id: string, needs: string[], description: string, argv: string[], inputs: string[], outputs: string[] };
export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };
export type TaskResult = { id: string, status: string, exit_code: number, input_sha256: string, output_sha256: string, detail: string };
export type State = { goal: string, tasks: Task[], order: string[], results: TaskResult[], blocked: string[], status: string, detail: string };
