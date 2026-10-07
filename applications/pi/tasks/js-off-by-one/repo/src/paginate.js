/** Split items into pages of `size`; pages are numbered from 1. */
export function paginate(items, page, size = 10) {
  if (!Number.isInteger(page) || page < 1) throw new RangeError('page must be a positive integer');
  const start = page * size;
  const slice = items.slice(start, start + size);
  return { page, pages: Math.max(1, Math.ceil(items.length / size)), items: slice, hasNext: start + size < items.length };
}
