import type { Section, SectionDigest } from '../types.js';

function collectSubtreeTexts(section: Section, byId: Map<string, string>): void {
  byId.set(section.id, section.text);
  for (const child of section.sections) {
    collectSubtreeTexts(child, byId);
  }
}

/**
 * Exact postcondition helper (auto-protected by `ensures:`):
 * verifies structural shape (id, title, ordered children) and guarantees every cited quote is non-empty,
 * references a section id within the subtree, and occurs verbatim in the text of the section it names.
 */
export default function citationsMatch(digest: SectionDigest, section: Section): boolean {
  if (digest.sectionId !== section.id || digest.title !== section.title) return false;
  if (!Array.isArray(digest.children) || digest.children.length !== section.sections.length) return false;
  if (!Array.isArray(digest.citations)) return false;

  const byId = new Map<string, string>();
  collectSubtreeTexts(section, byId);

  const quotesValid = digest.citations.every(citation => {
    if (typeof citation.quote !== 'string' || citation.quote.trim().length === 0) return false;
    if (!byId.has(citation.sectionId)) return false;
    return byId.get(citation.sectionId)!.includes(citation.quote);
  });
  if (!quotesValid) return false;

  return section.sections.every((child, index) => citationsMatch(digest.children[index]!, child));
}
