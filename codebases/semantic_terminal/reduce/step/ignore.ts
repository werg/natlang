export default function ignore(acc: Session, item: Event): Session {
return { ...acc, messages: [...acc.messages, `Ignored unknown event kind: ${item.kind}`] };
}
