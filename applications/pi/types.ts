/** How dangerous a shell command is in the user's project. */
export type Risk = 'safe' | 'review' | 'destructive';
/** Whether the agent's recent actions are getting anywhere. */
export type Progress = 'progressing' | 'repeating' | 'stuck';
/** Whether the agent's final answer actually completes the task. */
export type Completion = 'done' | 'unfinished';
/** Whether an edit's diff does what the agent said it would. */
export type EditCheck = 'as-intended' | 'unintended' | 'incomplete';
/** Whether the agent's next step can go to the small model. */
export type Route = 'routine' | 'hard';
/** One of edit's replacements: oldText must match a unique region of the original file exactly. */
export type Edit = { oldText: string, newText: string };
/** What pi's system prompt holds about a project. */
export type ProjectContext = {
  /** The AGENTS.md (or CLAUDE.md) files of the project directory and the directories above it, outermost first. */
  instructions: { path: string, content: string }[],
  /** Skills: when the task matches a description, read the file at location and follow it. */
  skills: { name: string, description: string, location: string }[],
  /** Files a quick scan suggests starting from (unverified), most likely first. */
  start: string[],
};
