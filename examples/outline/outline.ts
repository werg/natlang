/**
 * Digesting a nested document. `digest.nl` handles one section and calls itself on each subsection (it declares
 * `recursive: structural`), and its `ensures:` holds every digest to the section's shape and to its text.
 */
import digest from './digest.nl';
import type { Digest, Section } from './types.js';

/** The digest of a whole document, and its table of contents with one gist per section. */
export async function outline(document: Section): Promise<{ digest: Digest; contents: string[] }> {
  const result = await digest(document);
  const contents: string[] = [];
  const walk = (part: Digest, depth: number): void => {
    contents.push(`${'  '.repeat(depth)}${part.title}: ${part.gist}`);
    for (const child of part.parts) walk(child, depth + 1);
  };
  walk(result, 0);
  return { digest: result, contents };
}
