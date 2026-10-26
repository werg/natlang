export default function compareProtectedTokens(source: string, candidate: string, tokens: string[]): string[] {
  const findings: string[] = [];
  for (const token of tokens) {
    if (!token.length) throw new Error("empty_protected_token");
    const before = source.split(token).length - 1;
    const after = candidate.split(token).length - 1;
    if (before !== after) findings.push(token + ":" + before + ":" + after);
  }
  return findings;
}
