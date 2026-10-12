/** Wrap every case-insensitive occurrence of term in text with <mark></mark>. */
export function highlight(text, term) {
  if (!term) return text;
  const pattern = new RegExp(term, 'gi');
  return text.replace(pattern, (match) => `<mark>${match}</mark>`);
}

/** Number of case-insensitive occurrences of term in text. */
export function countMatches(text, term) {
  return (highlight(text, term).match(/<mark>/g) || []).length;
}
