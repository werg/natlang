export type Task = { id: Text, text: Text, done: Bool };
export type Board = { revision: Num, next_id: Num, items: Task[] };
export type UiEvent = { id: Text, kind: Text, value?: Text };
export type Decision = { kind: Text, item_id?: Text, text?: Text };
export type TaskGroup = { label: Text, ids: Text[] };
export type ViewPlan = { title: Text, summary: Text, groups: TaskGroup[] };
export type UiAction = { kind: Text, value?: Text, from?: Text };
export type UiNode = { tag: Text, text?: Text, value?: Text, id?: Text,
  label?: Text,
  placeholder?: Text, disabled?: Bool, action?: UiAction, children?: UiNode[] };
