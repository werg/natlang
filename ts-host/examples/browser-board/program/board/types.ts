export type Task = { id: string, text: string, done: boolean };
export type Board = { revision: number, next_id: number, items: Task[] };
export type UiEvent = { id: string, kind: string, value?: string };
export type Decision = { kind: string, item_id?: string, text?: string };
export type TaskGroup = { label: string, ids: string[] };
export type ViewPlan = { title: string, summary: string, groups: TaskGroup[] };
export type UiAction = { kind: string, value?: string, from?: string };
export type UiNode = { tag: string, text?: string, value?: string, id?: string,
  label?: string,
  placeholder?: string, disabled?: boolean, action?: UiAction, children?: UiNode[] };
