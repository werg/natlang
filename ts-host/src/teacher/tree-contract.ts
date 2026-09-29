/** Existing value slots specify their representation; new branch layouts need an independent ontology. */
export function hasExistingTreeValueContract(before: unknown, after: unknown): boolean {
  const valid = (value: any): boolean => value && typeof value.name === 'string' && Array.isArray(value.children) &&
    Object.keys(value).every(key => key === 'name' || key === 'children') &&
    new Set(value.children.map((child: any) => child?.name)).size === value.children.length;
  const compare = (a: any, b: any, parent?: string): boolean => {
    if (!valid(a) || !valid(b)) return false;
    if (!a.children.length && !b.children.length) return a.name === b.name || parent === 'equals' || parent === 'notEquals';
    if (a.name !== b.name || a.children.length !== b.children.length) return false;
    return a.children.every((child: any) => {
      // A literal may change its name under an existing single-valued comparator.
      if (a.children.length === 1 && ['equals', 'notEquals'].includes(a.name) && !child.children?.length && !b.children[0]?.children?.length)
        return compare(child, b.children[0], a.name);
      const target = b.children.find((candidate: any) => candidate?.name === child.name);
      return target !== undefined && compare(child, target, a.name);
    });
  };
  return compare(before, after);
}
