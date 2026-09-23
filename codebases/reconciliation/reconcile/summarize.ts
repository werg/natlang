import type { Customer, Event, Joined, Join, Label, Report } from "../types.js";
export default function summarize(joined: Join, labels: Label[]): Report {
if (joined.rows.length !== labels.length) throw new Error("row alignment lost");
const totals = Object.create(null), urgent = [], unmatched = [];
joined.rows.forEach((row, i) => {
  if (!row.matched) { unmatched.push(row.id); return; }
  totals[row.customer] = (totals[row.customer] || 0) + row.cents;
  if (labels[i] === "urgent") urgent.push(row.id);
});
return {totals, urgent, unmatched, duplicates: joined.duplicates};
}
