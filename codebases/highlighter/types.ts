export type SourceFile = { path: string, text: string };
export type Role = "signature" | "call" | "call_each" | "repeat" | "condition" | "exact" | "prose_step" | "return" | "comment" | "blank" | "leaf_text";
export type Parts = { frontmatter: string, lines: string[], functions: string[], is_code: boolean };
export type Highlighted = { path: string, roles: Role[], html: string };
