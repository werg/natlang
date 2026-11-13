/** "Did you mean": the candidate nearest to a misspelt name, when it is close enough to be the intended one. */
export function closestName(name: string, candidates: Iterable<string>): string | undefined {
  const limit = Math.max(1, Math.floor(name.length / 3));
  let best: string | undefined, bestDistance = limit + 1;
  for (const candidate of candidates) {
    if (candidate === name) continue;
    const distance = editDistance(name.toLowerCase(), candidate.toLowerCase());
    if (distance < bestDistance) { best = candidate; bestDistance = distance; }
  }
  return best;
}

/** Levenshtein distance with adjacent transpositions counted as one edit. */
function editDistance(left: string, right: string): number {
  const rows = Array.from({ length: left.length + 1 }, (_, index) => [index, ...new Array<number>(right.length).fill(0)]);
  for (let column = 1; column <= right.length; column++) rows[0]![column] = column;
  for (let row = 1; row <= left.length; row++) for (let column = 1; column <= right.length; column++) {
    const cost = left[row - 1] === right[column - 1] ? 0 : 1;
    let value = Math.min(rows[row - 1]![column]! + 1, rows[row]![column - 1]! + 1, rows[row - 1]![column - 1]! + cost);
    if (row > 1 && column > 1 && left[row - 1] === right[column - 2] && left[row - 2] === right[column - 1])
      value = Math.min(value, rows[row - 2]![column - 2]! + 1);
    rows[row]![column] = value;
  }
  return rows[left.length]![right.length]!;
}
