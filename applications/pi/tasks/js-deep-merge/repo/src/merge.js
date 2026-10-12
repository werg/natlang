/** True for {}-style objects (not arrays, not null). */
export function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Merge source into target recursively; values from source win. */
export function deepMerge(target, source) {
  for (const key of Object.keys(source)) {
    const value = source[key];
    if (isPlainObject(value) || Array.isArray(value)) {
      if (!(key in target)) target[key] = Array.isArray(value) ? [] : {};
      deepMerge(target[key], value);
    } else {
      target[key] = value;
    }
  }
  return target;
}
