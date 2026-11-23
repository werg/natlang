/** A job application as it arrives. The applicant's text is outside data: the runtime fences it and never follows it. */
export type Application = { id: string; text: Untrusted<string> };
/** A judgment of one application: the functions the screen applies. */
export type Check = (application: Application) => Promise<boolean>;
/** The shortlist and, for each requirement, the applications that miss it. */
export type Screening = { shortlisted: string[]; missing: Record<string, string[]> };
