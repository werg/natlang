export type Task = { id: string, needs: string[], description: string };
export type State = { tasks: Task[], order: string[], blocked: string[], finished: boolean };
