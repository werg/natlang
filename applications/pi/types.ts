/** How dangerous a shell command is in the user's project. */
export type Risk = 'safe' | 'review' | 'destructive';
/** Whether the agent's recent actions are getting anywhere. */
export type Progress = 'progressing' | 'repeating' | 'stuck';
/** Whether the agent's final answer actually completes the task. */
export type Completion = 'done' | 'unfinished';
/** Whether an edit's diff does what the agent said it would. */
export type EditCheck = 'as-intended' | 'unintended' | 'incomplete';
