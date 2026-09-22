export type Task = { id: Text, needs: Text[], description: Text, argv: Text[], inputs: Text[], outputs: Text[] };
export type File = { kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num };
export type TaskResult = { id: Text, status: Text, exit_code: Num, input_sha256: Text, output_sha256: Text, detail: Text };
export type State = { goal: Text, tasks: Task[], order: Text[], results: TaskResult[], blocked: Text[], status: Text, detail: Text };
