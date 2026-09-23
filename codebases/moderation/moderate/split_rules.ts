export default function split_rules(policy: string): string[] {
return policy.split("\n").map(l => l.replace(/^\s*(?:[-*]|\d+[.)])\s*/, "").trim()).filter(Boolean)
}
