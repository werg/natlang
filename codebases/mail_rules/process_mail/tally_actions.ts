export default function tally_actions(decisions: Decision[]): MailReport {
const n = a => decisions.filter(d => d.action === a).length
return { calendar: n("calendar"), reply_later: n("reply_later"), archived: n("archive") }
}
