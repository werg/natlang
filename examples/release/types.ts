/** One merged change as the tracker reports it. Its description is the author's text: outside data. */
export type Change = { id: string; title: string; description: Untrusted<string>; breaking: boolean };
/** Release notes: one line per change, each starting with the change's id. */
export type Notes = string[];
/** The release record that two maintainers edit at the same time. */
export type Release = { version: string; title: string; highlights: string; upgrade: string };
/** The paragraph at the top of the release page. */
export type Summary = Is<string, 'one paragraph for users of the project, about what changed for them'>;
