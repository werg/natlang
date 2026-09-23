export default function join_events(_customers: Customer[], events: Event[]): Join {
const customers = new Map(_customers.map(c => [c.id, c]));
const seen = new Set(); const rows = []; let duplicates = 0;
for (const e of events) {
  if (seen.has(e.id)) { duplicates++; continue; }
  seen.add(e.id); const c = customers.get(e.customer);
  rows.push({...e, matched: !!c, tier: c ? c.tier : "unknown"});
}
return {rows, duplicates};
}
