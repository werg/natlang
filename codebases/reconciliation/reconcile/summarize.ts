/*---
args:
  joined: Join
  labels: Label[]
returns: Report
---*/
if (args.joined.rows.length !== args.labels.length) throw new Error("row alignment lost");
const totals = Object.create(null), urgent = [], unmatched = [];
args.joined.rows.forEach((row, i) => {
  if (!row.matched) { unmatched.push(row.id); return; }
  totals[row.customer] = (totals[row.customer] || 0) + row.cents;
  if (args.labels[i] === "urgent") urgent.push(row.id);
});
return {totals, urgent, unmatched, duplicates: args.joined.duplicates};
