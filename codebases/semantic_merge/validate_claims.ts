import type { State, Node, Edge, Item, Rule, Object, Event } from "./types.js";
export default function validate_claims(updates: Update[], applied: string[], alternatives: Alternative[]): boolean {
const ids = updates.map(u => u.id);
const claimed = [...applied, ...alternatives.flatMap(a => a.update_ids)];
return claimed.length === ids.length && new Set(claimed).size === ids.length &&
       claimed.every(id => ids.includes(id)) &&
       alternatives.every(a => a.update_ids.length > 0 && a.reason.trim().length > 0);
}
