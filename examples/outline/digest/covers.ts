import type { Digest, Section } from '../types.js';

/** Whether a digest has the section's shape: the same title, and one part per subsection with its title, in order. */
export default function covers(digest: Digest, section: Section): boolean {
  return digest.title === section.title && digest.parts.length === section.sections.length &&
    section.sections.every((child, index) => digest.parts[index]!.title === child.title);
}
