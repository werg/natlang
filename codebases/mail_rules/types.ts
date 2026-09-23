export type Decision = { action: "calendar" | "reply_later" | "archive", date: string };
export type MailReport = { calendar: number, reply_later: number, archived: number };
