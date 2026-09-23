import type { Event, File, Recipe, Job, Outcome, Session, ViewBlock, TerminalView } from "../types.js";

export default function ignore(acc: Session, item: Event): Session {
return { ...acc, messages: [...acc.messages, `Ignored unknown event kind: ${item.kind}`] };
}
