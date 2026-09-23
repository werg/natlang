import { decisionType } from './contracts.mjs';
/** Small application helpers. No runtime primitives and no hidden model calls. */
export const clone = value => structuredClone(value);
export const requireValue = (test, message) => { if (!test)
    throw new Error(message); };
export const text = (value, name = 'text') => { requireValue(typeof value === 'string' && value.trim(), `${name} is required`); return value.trim(); };
export const number = (value, name, min = -Infinity, max = Infinity) => {
    const n = Number(value);
    requireValue(Number.isFinite(n) && n >= min && n <= max, `${name} must be between ${min} and ${max}`);
    return n;
};
export const find = (rows, id) => { const row = rows.find(row => row.id === id); requireValue(row, `Unknown item: ${id}`); return row; };
export const uid = rows => `item-${rows.reduce((max, row) => Math.max(max, Number(String(row.id).split('-').at(-1)) || 0), 0) + 1}`;
export const json = value => JSON.stringify(value, null, 2);
export const action = (label, kind, data = {}, tone = '') => ({ label, kind, data, tone });
export const field = (name, label, value = '', type = 'text', options) => ({ name, label, value, type, ...(options ? { options } : {}) });
export const panel = (id, title, kind, extra = {}) => ({ id, title, kind, ...extra });
export const form = (id, title, fields, actions, extra = {}) => panel(id, title, 'form', { fields, actions, ...extra });
export const table = (id, title, rows, columns, extra = {}) => panel(id, title, 'table', { rows, columns, ...extra });
export const define = spec => ({ ...spec, version: 1, decisionType: spec.decisionType ?? decisionType(spec.id) });
export function revised(state, note) { state.revision++; state.notice = note; return state; }
export function unsupported(action) { throw new Error(`Unsupported action: ${action}`); }
export function parseRows(source) {
    const rows = JSON.parse(source);
    requireValue(Array.isArray(rows) && rows.every(row => row && typeof row === 'object' && !Array.isArray(row)), 'Use a JSON array of records');
    return rows;
}
export const COMMON = 'revision: number; notice: string;';
export const initial = data => ({ revision: 0, notice: 'Ready when you are.', ...data });
export function rng(seed) { let value = seed >>> 0; return () => { value = (Math.imul(value, 1664525) + 1013904223) >>> 0; return value / 4294967296; }; }
/** Portable structural equality; record key order is not part of a natlang value. */
export function sameValue(a, b) {
    if (a === b)
        return true;
    if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b))
        return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every(key => Object.hasOwn(b, key) && sameValue(a[key], b[key]));
}

/** A bounded presentation sample, never a replacement for the host-owned value. */
export function valuePreview(value,depth=0){
 if(typeof value==='string')return value.length>256?value.slice(0,256)+'…':value;
 if(value&&typeof value==='object'){
  if(depth>=4)return Array.isArray(value)?`[${value.length} values]`:'[record]';
  if(Array.isArray(value))return value.slice(0,8).map(row=>valuePreview(row,depth+1));
  return Object.fromEntries(Object.entries(value).slice(0,16).map(([key,row])=>[key,valuePreview(row,depth+1)]));
 }
 return value;
}
