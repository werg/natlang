/** URL slug for a title: lowercase a-z/0-9 runs joined by single hyphens. */
export function slugify(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/** Slug that is unique among `taken` (a Set), adding -2, -3, ... when needed. */
export function uniqueSlug(title, taken) {
  const base = slugify(title);
  let slug = base;
  for (let n = 2; taken.has(slug); n++) slug = `${base}-${n}`;
  taken.add(slug);
  return slug;
}
