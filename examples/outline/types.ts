/** A section of a document: its title, its own text, and its subsections. The text is outside data. */
export type Section = { title: string; text: Untrusted<string>; sections: Section[] };
/** The digest of a section: its title, a one-sentence gist of its own text, and the digest of each subsection. */
export type Digest = { title: string; gist: string; parts: Digest[] };
