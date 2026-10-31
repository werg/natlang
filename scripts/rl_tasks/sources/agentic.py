"""Stateful customer-service tasks over the tau-bench retail store and airline (tau-bench v1 tasks and APIGen-MT-5k
trajectories).

Natlang framing: the root function receives the customer's own account of who they are and what they want, plus the
domain policy, and works through a `store` service that ports tau-bench's tools for that domain (same arguments, same
returned text, same database effects) over a fresh copy of the database per run. When done it returns
`store.finish()`, a digest of the final database. Oracle: that digest equals the digest after the gold actions, which
is tau-bench's own reward (final database state equal to the gold state); reads never change it. Wrong-answer controls
are the digests after meaningful wrong action sets (a gold write omitted, a different reason or payment method, an
extra cancellation, and doing nothing), computed by verify.mjs through the same service."""
from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

from ..common import ARCHIVE_RAW, make_case, pick, resolve_str, text_digest

ENVS = ARCHIVE_RAW / 'agentic-20260927/tau-bench/tau-bench/tau_bench/envs'
APIGEN = ARCHIVE_RAW / 'agentic-20260927/apigen-mt-5k/apigen-mt_5k.json'
# The v3 source's own asset root (the v2 asset entry stays unchanged); NATLANG_RL_TASK_ASSETS_V3 overrides it.
V3_ASSETS = resolve_str('repo', 'runs', 'natlang-rl-broad-external-20261031-v3', 'assets')
BUILD_V3_ASSETS = resolve_str('data_nvme', 'broad-rl-tasks-20261031', 'assets-v3')
DOMAINS = {
    'retail': {
        'tables': ('users', 'orders', 'products'),
        'writes': ['cancel_pending_order', 'exchange_delivered_order_items', 'modify_pending_order_address',
                   'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address',
                   'return_delivered_order_items'],
        'reads': ['find_user_id_by_email', 'find_user_id_by_name_zip', 'get_user_details', 'get_order_details',
                  'get_product_details', 'list_all_product_types', 'calculate', 'transfer_to_human_agents'],
        'record': 'orders', 'record_key': 'order_id', 'held_files': ('tasks_test.py', 'tasks_dev.py'),
        'task_files': ('tasks_train.py', 'tasks_dev.py', 'tasks_test.py'),
    },
    'airline': {
        'tables': ('flights', 'reservations', 'users'),
        'writes': ['book_reservation', 'cancel_reservation', 'send_certificate', 'update_reservation_baggages',
                   'update_reservation_flights', 'update_reservation_passengers'],
        'reads': ['get_user_details', 'get_reservation_details', 'list_all_airports', 'search_direct_flight',
                  'search_onestop_flight', 'calculate', 'transfer_to_human_agents'],
        'record': 'reservations', 'record_key': 'reservation_id', 'held_files': ('tasks_test.py',),
        'task_files': ('tasks_test.py',),
    },
}

SERVICE_HEAD = r'''import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
const FILE = __FILE__;
const SHA256 = __SHA__;
type Db = Record<string, Record<string, any>>;
function pristine(): string {
  const cache = ((globalThis as any).__natlangRlTauStores ??= new Map());
  if (!cache.has(SHA256)) {
    const bytes = readFileSync((process.env.NATLANG_RL_TASK_ASSETS_V3 ?? __ROOT__) + '/' + FILE);
    if (createHash('sha256').update(bytes).digest('hex') !== SHA256) throw new Error('store asset changed: ' + FILE);
    cache.set(SHA256, bytes.toString('utf8'));
  }
  return cache.get(SHA256);
}
let state: Db | undefined;
const db = (): Db => (state ??= JSON.parse(pristine()));
const round2 = (x: number) => Math.round((x + Number.EPSILON) * 100) / 100;
const count = (xs: string[], x: string) => xs.filter(y => y === x).length;
function canon(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon((value as any)[k])]));
  return value;
}
'''
RETAIL_TS = r'''__DOCS_find_user_id_by_email__
export function find_user_id_by_email(args: { email: string }): string {
  for (const [id, user] of Object.entries(db().users)) if (String(user.email).toLowerCase() === String(args.email).toLowerCase()) return id;
  return 'Error: user not found';
}
__DOCS_find_user_id_by_name_zip__
export function find_user_id_by_name_zip(args: { first_name: string; last_name: string; zip: string }): string {
  for (const [id, user] of Object.entries(db().users))
    if (user.name.first_name.toLowerCase() === String(args.first_name).toLowerCase() &&
        user.name.last_name.toLowerCase() === String(args.last_name).toLowerCase() && user.address.zip === args.zip) return id;
  return 'Error: user not found';
}
__DOCS_get_user_details__
export function get_user_details(args: { user_id: string }): string {
  const user = db().users[args.user_id];
  return user ? JSON.stringify(user) : 'Error: user not found';
}
__DOCS_get_order_details__
export function get_order_details(args: { order_id: string }): string {
  const order = db().orders[args.order_id];
  return order ? JSON.stringify(order) : 'Error: order not found';
}
__DOCS_get_product_details__
export function get_product_details(args: { product_id: string }): string {
  const product = db().products[args.product_id];
  return product ? JSON.stringify(product) : 'Error: product not found';
}
__DOCS_list_all_product_types__
export function list_all_product_types(): string {
  const names: Record<string, string> = {};
  for (const product of Object.values(db().products)) names[product.name] = product.product_id;
  return JSON.stringify(Object.fromEntries(Object.entries(names).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))));
}
__DOCS_calculate__
export function calculate(args: { expression: string }): string {
  if (![...String(args.expression)].every(ch => '0123456789+-*/(). '.includes(ch))) return 'Error: invalid characters in expression';
  try { const value = Number(new Function(`return (${args.expression});`)()); return Number.isFinite(value) ? String(round2(value)) : 'Error: invalid expression'; }
  catch (error) { return `Error: ${(error as Error).message}`; }
}
__DOCS_transfer_to_human_agents__
export function transfer_to_human_agents(args: { summary: string }): string { return 'Transfer successful'; }
__DOCS_cancel_pending_order__
export function cancel_pending_order(args: { order_id: string; reason: string }): string {
  const data = db(), order = data.orders[args.order_id];
  if (!order) return 'Error: order not found';
  if (order.status !== 'pending') return 'Error: non-pending order cannot be cancelled';
  if (!['no longer needed', 'ordered by mistake'].includes(args.reason)) return 'Error: invalid reason';
  const refunds = [];
  for (const payment of order.payment_history) {
    refunds.push({ transaction_type: 'refund', amount: payment.amount, payment_method_id: payment.payment_method_id });
    if (payment.payment_method_id.includes('gift_card')) {
      const method = data.users[order.user_id].payment_methods[payment.payment_method_id];
      method.balance = round2(method.balance + payment.amount);
    }
  }
  order.status = 'cancelled';
  order.cancel_reason = args.reason;
  order.payment_history.push(...refunds);
  return JSON.stringify(order);
}
function priceChange(order: any, itemIds: string[], newItemIds: string[]): number | string {
  const products = db().products, all = order.items.map((item: any) => item.item_id);
  for (const id of itemIds) if (count(itemIds, id) > count(all, id)) return `Error: ${id} not found`;
  if (itemIds.length !== newItemIds.length) return 'Error: the number of items to be exchanged should match';
  let diff = 0;
  for (let i = 0; i < itemIds.length; i++) {
    const item = order.items.find((x: any) => x.item_id === itemIds[i]);
    const variant = products[item.product_id].variants[newItemIds[i]!];
    if (!variant || !variant.available) return `Error: new item ${newItemIds[i]} not found or available`;
    diff += variant.price - item.price;
  }
  return diff;
}
__DOCS_exchange_delivered_order_items__
export function exchange_delivered_order_items(args: { order_id: string; item_ids: string[]; new_item_ids: string[]; payment_method_id: string }): string {
  const data = db(), order = data.orders[args.order_id];
  if (!order) return 'Error: order not found';
  if (order.status !== 'delivered') return 'Error: non-delivered order cannot be exchanged';
  const change = priceChange(order, args.item_ids, args.new_item_ids);
  if (typeof change === 'string') return change;
  const diff = round2(change);
  const method = data.users[order.user_id].payment_methods[args.payment_method_id];
  if (!method) return 'Error: payment method not found';
  if (method.source === 'gift_card' && method.balance < diff) return 'Error: insufficient gift card balance to pay for the price difference';
  order.status = 'exchange requested';
  order.exchange_items = [...args.item_ids].sort();
  order.exchange_new_items = [...args.new_item_ids].sort();
  order.exchange_payment_method_id = args.payment_method_id;
  order.exchange_price_difference = diff;
  return JSON.stringify(order);
}
__DOCS_modify_pending_order_address__
export function modify_pending_order_address(args: { order_id: string; address1: string; address2: string; city: string; state: string; country: string; zip: string }): string {
  const order = db().orders[args.order_id];
  if (!order) return 'Error: order not found';
  if (order.status !== 'pending') return 'Error: non-pending order cannot be modified';
  order.address = { address1: args.address1, address2: args.address2, city: args.city, state: args.state, country: args.country, zip: args.zip };
  return JSON.stringify(order);
}
__DOCS_modify_pending_order_items__
export function modify_pending_order_items(args: { order_id: string; item_ids: string[]; new_item_ids: string[]; payment_method_id: string }): string {
  const data = db(), order = data.orders[args.order_id];
  if (!order) return 'Error: order not found';
  if (order.status !== 'pending') return 'Error: non-pending order cannot be modified';
  const diff = priceChange(order, args.item_ids, args.new_item_ids);
  if (typeof diff === 'string') return diff;
  const method = data.users[order.user_id].payment_methods[args.payment_method_id];
  if (!method) return 'Error: payment method not found';
  if (method.source === 'gift_card' && method.balance < diff) return 'Error: insufficient gift card balance to pay for the new item';
  order.payment_history.push({ transaction_type: diff > 0 ? 'payment' : 'refund', amount: Math.abs(diff), payment_method_id: args.payment_method_id });
  if (method.source === 'gift_card') method.balance = round2(method.balance - diff);
  for (let i = 0; i < args.item_ids.length; i++) {
    const item = order.items.find((x: any) => x.item_id === args.item_ids[i]);
    const variant = data.products[item.product_id].variants[args.new_item_ids[i]!];
    item.item_id = args.new_item_ids[i]; item.price = variant.price; item.options = variant.options;
  }
  order.status = 'pending (item modified)';
  return JSON.stringify(order);
}
__DOCS_modify_pending_order_payment__
export function modify_pending_order_payment(args: { order_id: string; payment_method_id: string }): string {
  const data = db(), order = data.orders[args.order_id];
  if (!order) return 'Error: order not found';
  if (order.status !== 'pending') return 'Error: non-pending order cannot be modified';
  const methods = data.users[order.user_id].payment_methods;
  if (!methods[args.payment_method_id]) return 'Error: payment method not found';
  const first = order.payment_history[0];
  if (order.payment_history.length > 1 || first.transaction_type !== 'payment') return 'Error: there should be exactly one payment for a pending order';
  if (first.payment_method_id === args.payment_method_id) return 'Error: the new payment method should be different from the current one';
  const amount = first.amount, method = methods[args.payment_method_id];
  if (method.source === 'gift_card' && method.balance < amount) return 'Error: insufficient gift card balance to pay for the order';
  order.payment_history.push({ transaction_type: 'payment', amount, payment_method_id: args.payment_method_id },
    { transaction_type: 'refund', amount, payment_method_id: first.payment_method_id });
  if (method.source === 'gift_card') method.balance = round2(method.balance - amount);
  if (first.payment_method_id.includes('gift_card')) {
    const old = methods[first.payment_method_id];
    old.balance = round2(old.balance + amount);
  }
  return JSON.stringify(order);
}
__DOCS_modify_user_address__
export function modify_user_address(args: { user_id: string; address1: string; address2: string; city: string; state: string; country: string; zip: string }): string {
  const user = db().users[args.user_id];
  if (!user) return 'Error: user not found';
  user.address = { address1: args.address1, address2: args.address2, city: args.city, state: args.state, country: args.country, zip: args.zip };
  return JSON.stringify(user);
}
__DOCS_return_delivered_order_items__
export function return_delivered_order_items(args: { order_id: string; item_ids: string[]; payment_method_id: string }): string {
  const data = db(), order = data.orders[args.order_id];
  if (!order) return 'Error: order not found';
  if (order.status !== 'delivered') return 'Error: non-delivered order cannot be returned';
  if (!data.users[order.user_id].payment_methods[args.payment_method_id]) return 'Error: payment method not found';
  if (!args.payment_method_id.includes('gift_card') && args.payment_method_id !== order.payment_history[0].payment_method_id)
    return 'Error: payment method should be either the original payment method or a gift card';
  const all = order.items.map((item: any) => item.item_id);
  for (const id of args.item_ids) if (count(args.item_ids, id) > count(all, id)) return 'Error: some item not found';
  order.status = 'return requested';
  order.return_items = [...args.item_ids].sort();
  order.return_payment_method_id = args.payment_method_id;
  return JSON.stringify(order);
}
'''
AIRLINE_TS = r'''__DOCS_get_user_details__
export function get_user_details(args: { user_id: string }): string {
  const user = db().users[args.user_id];
  return user ? JSON.stringify(user) : 'Error: user not found';
}
__DOCS_get_reservation_details__
export function get_reservation_details(args: { reservation_id: string }): string {
  const reservation = db().reservations[args.reservation_id];
  return reservation ? JSON.stringify(reservation) : 'Error: user not found';
}
__DOCS_list_all_airports__
export function list_all_airports(): string {
  const airports = ['SFO', 'JFK', 'LAX', 'ORD', 'DFW', 'DEN', 'SEA', 'ATL', 'MIA', 'BOS', 'PHX', 'IAH', 'LAS', 'MCO', 'EWR', 'CLT', 'MSP', 'DTW', 'PHL', 'LGA'];
  const cities = ['San Francisco', 'New York', 'Los Angeles', 'Chicago', 'Dallas', 'Denver', 'Seattle', 'Atlanta', 'Miami', 'Boston', 'Phoenix', 'Houston', 'Las Vegas', 'Orlando', 'Newark', 'Charlotte', 'Minneapolis', 'Detroit', 'Philadelphia', 'LaGuardia'];
  return JSON.stringify(Object.fromEntries(airports.map((a, i) => [a, cities[i]])));
}
const withoutDates = (flight: any) => Object.fromEntries(Object.entries(flight).filter(([k]) => k !== 'dates'));
__DOCS_search_direct_flight__
export function search_direct_flight(args: { origin: string; destination: string; date: string }): string {
  const results = [];
  for (const flight of Object.values(db().flights))
    if (flight.origin === args.origin && flight.destination === args.destination && flight.dates[args.date]?.status === 'available')
      results.push({ ...withoutDates(flight), ...flight.dates[args.date] });
  return JSON.stringify(results);
}
__DOCS_search_onestop_flight__
export function search_onestop_flight(args: { origin: string; destination: string; date: string }): string {
  const flights = Object.values(db().flights), results = [];
  for (const first of flights) {
    if (first.origin !== args.origin) continue;
    for (const second of flights) {
      if (second.destination !== args.destination || first.destination !== second.origin) continue;
      const date2 = String(first.scheduled_arrival_time_est).includes('+1') ? `2024-05-${Number(args.date.slice(-2)) + 1}` : args.date;
      if (first.scheduled_arrival_time_est > second.scheduled_departure_time_est) continue;
      if (first.dates[args.date]?.status === 'available' && second.dates[date2]?.status === 'available')
        results.push([{ ...withoutDates(first), ...first.dates[args.date], date: args.date },
          { ...withoutDates(second), ...second.dates[args.date], date: date2 }]);
    }
  }
  return JSON.stringify(results);
}
__DOCS_calculate__
export function calculate(args: { expression: string }): string {
  if (![...String(args.expression)].every(ch => '0123456789+-*/(). '.includes(ch))) return 'Error: invalid characters in expression';
  try { const value = Number(new Function(`return (${args.expression});`)()); return Number.isFinite(value) ? String(round2(value)) : 'Error: invalid expression'; }
  catch (error) { return `Error: ${(error as Error).message}`; }
}
__DOCS_transfer_to_human_agents__
export function transfer_to_human_agents(args: { summary: string }): string { return 'Transfer successful'; }
function pricedFlight(flight: any, cabin: string, passengers: number): string | undefined {
  const data = db().flights[flight.flight_number];
  if (!data) return `Error: flight ${flight.flight_number} not found`;
  const day = data.dates[flight.date];
  if (!day) return `Error: flight ${flight.flight_number} not found on date ${flight.date}`;
  if (day.status !== 'available') return `Error: flight ${flight.flight_number} not available on date ${flight.date}`;
  if (day.available_seats[cabin] < passengers) return `Error: not enough seats on flight ${flight.flight_number}`;
  flight.price = day.prices[cabin]; flight.origin = data.origin; flight.destination = data.destination;
  return undefined;
}
__DOCS_book_reservation__
export function book_reservation(args: { user_id: string; origin: string; destination: string; flight_type: string; cabin: string;
    flights: { flight_number: string; date: string }[]; passengers: { first_name: string; last_name: string; dob: string }[];
    payment_methods: { payment_id: string; amount: number }[]; total_baggages: number; nonfree_baggages: number; insurance: string }): string {
  const data = db(), user = data.users[args.user_id];
  if (!user) return 'Error: user not found';
  let id = 'HATHAT';
  if (data.reservations[id]) { id = 'HATHAU'; if (data.reservations[id]) id = 'HATHAV'; }
  const reservation: any = { reservation_id: id, user_id: args.user_id, origin: args.origin, destination: args.destination,
    flight_type: args.flight_type, cabin: args.cabin, flights: structuredClone(args.flights), passengers: args.passengers,
    payment_history: args.payment_methods, created_at: '2024-05-15T15:00:00', total_baggages: args.total_baggages,
    nonfree_baggages: args.nonfree_baggages, insurance: args.insurance };
  let total = 0;
  for (const flight of reservation.flights) {
    const error = pricedFlight(flight, args.cabin, args.passengers.length);
    if (error) return error;
    total += flight.price * args.passengers.length;
  }
  if (args.insurance === 'yes') total += 30 * args.passengers.length;
  total += 50 * args.nonfree_baggages;
  for (const payment of args.payment_methods) {
    const method = user.payment_methods[payment.payment_id];
    if (!method) return `Error: payment method ${payment.payment_id} not found`;
    if (['gift_card', 'certificate'].includes(method.source) && method.amount < payment.amount)
      return `Error: not enough balance in payment method ${payment.payment_id}`;
  }
  const paid = args.payment_methods.reduce((sum, p) => sum + p.amount, 0);
  if (paid !== total) return `Error: payment amount does not add up, total price is ${total}, but paid ${paid}`;
  for (const payment of args.payment_methods) {
    const method = user.payment_methods[payment.payment_id];
    if (method.source === 'gift_card') method.amount -= payment.amount;
    else if (method.source === 'certificate') delete user.payment_methods[payment.payment_id];
  }
  data.reservations[id] = reservation;
  user.reservations.push(id);
  return JSON.stringify(reservation);
}
__DOCS_cancel_reservation__
export function cancel_reservation(args: { reservation_id: string }): string {
  const reservation = db().reservations[args.reservation_id];
  if (!reservation) return 'Error: reservation not found';
  const refunds = reservation.payment_history.map((p: any) => ({ payment_id: p.payment_id, amount: -p.amount }));
  reservation.payment_history.push(...refunds);
  reservation.status = 'cancelled';
  return JSON.stringify(reservation);
}
__DOCS_send_certificate__
export function send_certificate(args: { user_id: string; amount: number }): string {
  const user = db().users[args.user_id];
  if (!user) return 'Error: user not found';
  for (const n of [3221322, 3221323, 3221324]) {
    const id = `certificate_${n}`;
    if (!user.payment_methods[id]) {
      user.payment_methods[id] = { source: 'certificate', amount: args.amount, id };
      return `Certificate ${id} added to user ${args.user_id} with amount ${args.amount}.`;
    }
  }
  return 'None';
}
function chargeable(reservation: any, paymentId: string, total: number): any {
  const method = db().users[reservation.user_id].payment_methods[paymentId];
  if (!method) return 'Error: payment method not found';
  if (method.source === 'certificate') return 'Error: certificate cannot be used to update reservation';
  if (method.source === 'gift_card' && method.amount < total) return 'Error: gift card balance is not enough';
  return method;
}
__DOCS_update_reservation_baggages__
export function update_reservation_baggages(args: { reservation_id: string; total_baggages: number; nonfree_baggages: number; payment_id: string }): string {
  const reservation = db().reservations[args.reservation_id];
  if (!reservation) return 'Error: reservation not found';
  const total = 50 * Math.max(0, args.nonfree_baggages - reservation.nonfree_baggages);
  const method = chargeable(reservation, args.payment_id, total);
  if (typeof method === 'string') return method;
  reservation.total_baggages = args.total_baggages;
  reservation.nonfree_baggages = args.nonfree_baggages;
  if (method.source === 'gift_card') method.amount -= total;
  if (total !== 0) reservation.payment_history.push({ payment_id: args.payment_id, amount: total });
  return JSON.stringify(reservation);
}
__DOCS_update_reservation_flights__
export function update_reservation_flights(args: { reservation_id: string; cabin: string; flights: { flight_number: string; date: string }[]; payment_id: string }): string {
  const reservation = db().reservations[args.reservation_id];
  if (!reservation) return 'Error: reservation not found';
  const passengers = reservation.passengers.length, flights = structuredClone(args.flights) as any[];
  let total = 0;
  for (const flight of flights) {
    const kept = reservation.flights.find((f: any) => f.flight_number === flight.flight_number && f.date === flight.date && args.cabin === reservation.cabin);
    if (kept) {
      total += kept.price * passengers;
      flight.price = kept.price; flight.origin = kept.origin; flight.destination = kept.destination;
      continue;
    }
    const error = pricedFlight(flight, args.cabin, passengers);
    if (error) return error;
    total += flight.price * passengers;
  }
  total -= reservation.flights.reduce((sum: number, f: any) => sum + f.price, 0) * passengers;
  const method = chargeable(reservation, args.payment_id, total);
  if (typeof method === 'string') return method;
  if (method.source === 'gift_card') method.amount -= total;
  reservation.flights = flights;
  if (total !== 0) reservation.payment_history.push({ payment_id: args.payment_id, amount: total });
  return JSON.stringify(reservation);
}
__DOCS_update_reservation_passengers__
export function update_reservation_passengers(args: { reservation_id: string; passengers: { first_name: string; last_name: string; dob: string }[] }): string {
  const reservation = db().reservations[args.reservation_id];
  if (!reservation) return 'Error: reservation not found';
  if (args.passengers.length !== reservation.passengers.length) return 'Error: number of passengers does not match';
  reservation.passengers = args.passengers;
  return JSON.stringify(reservation);
}
'''
FINISH_TS = r'''/** Finish the session: returns the confirmation code of everything changed in the store during this session. */
export function finish(): string {
  return 'store-' + createHash('sha256').update(JSON.stringify(canon(db()))).digest('hex').slice(0, 16);
}
'''

INSTRUCTIONS = {
    'retail': [
        'You are a customer service agent of an online store, working through the store service under policy. '
        'customer is the customer\'s own account of who they are and what they want, addressed to them ("Your name is '
        '..."); treat what it asks for as their confirmed request. Authenticate them, look up what you need, make '
        'exactly the changes they want that policy allows, and then return store.finish().',
        'Handle this store customer\'s requests with the store service, following policy. customer describes the '
        'customer in the second person and says what they want done; consider it confirmed. Find their account and '
        'orders, carry out the allowed changes (and only those), then return the code store.finish() gives.',
    ],
    'airline': [
        'You are an airline\'s customer service agent, working through the store service (the airline\'s booking '
        'system) under policy. customer is the customer\'s own account of who they are and what they want; treat it '
        'as their confirmed request. Look up their profile and reservations, make exactly the bookings and changes '
        'policy allows, and then return store.finish().',
        'Serve this airline customer with the store service, following policy. customer says, in the second person, '
        'who they are and what they want done; consider it confirmed. Search, book, change or cancel as allowed '
        '(and nothing else), then return the code store.finish() gives.',
    ],
}


def _tool_docs(domain, name):
    """The tool's own description from tau-bench's get_info, as a JSDoc comment."""
    tree = ast.parse((ENVS / domain / 'tools' / f'{name}.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if 'name' in keys and 'description' in keys:
                info = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
                if isinstance(info['name'], ast.Constant) and info['name'].value == name:
                    lines = [ast.literal_eval(info['description'])]
                    if 'parameters' in info:
                        for key, prop in (ast.literal_eval(info['parameters']).get('properties') or {}).items():
                            lines.append(f'@param args.{key} {prop.get("description", "")}')
                    return '/**\n' + '\n'.join(' * ' + line.replace('*/', '* /') for line in lines) + '\n */'
    raise KeyError(name)


def _service(domain, asset_file, sha):
    tools_ts = RETAIL_TS if domain == 'retail' else AIRLINE_TS
    code = (SERVICE_HEAD + tools_ts + FINISH_TS).replace('__FILE__', json.dumps(asset_file)) \
        .replace('__SHA__', json.dumps(sha)).replace('__ROOT__', json.dumps(V3_ASSETS))
    spec = DOMAINS[domain]
    for name in spec['reads'] + spec['writes']:
        code = code.replace(f'__DOCS_{name}__', _tool_docs(domain, name))
    assert '__DOCS_' not in code, 'undocumented tool'
    return code


def _db_asset(domain, out_assets):
    """The domain database as one JSON asset, bound by SHA-256."""
    data = {key: json.loads((ENVS / domain / 'data' / f'{key}.json').read_text()) for key in DOMAINS[domain]['tables']}
    rel = f'stores/tau-{domain}-db.json'
    target = Path(out_assets) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    if not target.exists() or target.read_text() != text:
        target.write_text(text)
    return rel, hashlib.sha256(target.read_bytes()).hexdigest(), data


def _tasks(path):
    """tau-bench Task(...) literals of a tasks file, parsed without importing it."""
    out = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'Task':
            kw = {k.arg: k.value for k in node.keywords}
            actions = [{k.arg: ast.literal_eval(k.value) for k in a.keywords} for a in kw['actions'].elts]
            out.append({'user_id': ast.literal_eval(kw['user_id']), 'instruction': ast.literal_eval(kw['instruction']),
                        'actions': actions, 'outputs': ast.literal_eval(kw['outputs'])})
    return out


def _users(domain, actions, data):
    spec = DOMAINS[domain]
    users = {a['kwargs'].get('user_id') for a in actions if isinstance(a['kwargs'].get('user_id'), str)}
    for a in actions:
        record = data[spec['record']].get(str(a['kwargs'].get(spec['record_key'], '')), {})
        if record.get('user_id'):
            users.add(record['user_id'])
    return users


def _wrong_action_sets(domain, actions, data):
    """Meaningful wrong attempts: drop the last gold write, swap a cancellation reason, pay with another method, or
    cancel a different open order/reservation of the same user. (Doing nothing is added by verify.mjs.)"""
    spec = DOMAINS[domain]
    writes = [i for i, a in enumerate(actions) if a['name'] in spec['writes']]
    sets = []
    if writes:
        sets.append([a for i, a in enumerate(actions) if i != writes[-1]])
        last = actions[writes[-1]]
        kwargs = dict(last['kwargs'])
        swap = None
        if last['name'] == 'cancel_pending_order':
            kwargs['reason'] = 'ordered by mistake' if kwargs.get('reason') == 'no longer needed' else 'no longer needed'
            swap = kwargs
        for field in ('payment_method_id', 'payment_id'):
            if field in kwargs:
                users = sorted(_users(domain, actions, data))
                methods = list(data['users'].get(users[0], {}).get('payment_methods', {})) if users else []
                other = next((m for m in methods if m != kwargs[field]), None)
                if other:
                    swap = {**kwargs, field: other}
        if swap:
            sets.append(actions[:writes[-1]] + [{'name': last['name'], 'kwargs': swap}] + actions[writes[-1] + 1:])
    users = sorted(_users(domain, actions, data))
    touched = {str(a['kwargs'].get(spec['record_key'])) for a in actions}
    if users:
        for record_id in data['users'].get(users[0], {}).get(spec['record'], []) or []:
            record = data[spec['record']].get(record_id, {})
            if record_id in touched:
                continue
            if domain == 'retail' and record.get('status') == 'pending':
                sets.append(actions + [{'name': 'cancel_pending_order', 'kwargs': {'order_id': record_id, 'reason': 'no longer needed'}}])
                break
            if domain == 'airline' and record.get('status') != 'cancelled':
                sets.append(actions + [{'name': 'cancel_reservation', 'kwargs': {'reservation_id': record_id}}])
                break
    return sets


def _reference(actions):
    lines = [f'await store.{a["name"]}({json.dumps(a["kwargs"])});' for a in actions]
    return [['eval', {'code': '\n'.join(lines + ['return store.finish();']), 'finish': True}]]


def _case(*, domain, source, upstream, item, unit, split, customer, actions, policy, service, data, license_, key,
          revision):
    writes = [a for a in actions if a['name'] in DOMAINS[domain]['writes']]
    case = make_case(
        source=source, family=f'{domain}_service', shape=f'{source}-{min(len(writes), 3)}writes',
        upstream=upstream, upstream_id=item, upstream_split=split, unit=unit, groups=[unit],
        root_name='serve_customer', args={'customer': 'string', 'policy': 'string'}, returns='string',
        instruction=pick(INSTRUCTIONS[domain], key), inputs={'customer': customer, 'policy': policy}, expected=None,
        oracle='exact', level=min(3, 1 + len(writes)), license_=license_, services={'store': service},
        domain='actor', slice_='inline_placement', reference_root=_reference(actions), revision=revision)
    case['curriculum']['external']['call_sets'] = {'service': 'store', 'mode': 'state', 'finish': 'finish',
                                                   'gold': actions, 'wrong': _wrong_action_sets(domain, actions, data),
                                                   'alias': []}
    return case


def _held_users(domain):
    return {t['user_id'] for name in DOMAINS[domain]['held_files'] for t in _tasks(ENVS / domain / name)}


def tau(domain, out_assets=BUILD_V3_ASSETS):
    """tau-bench v1 tasks: train tasks are train; test and dev tasks (whose revised versions are tau2's tasks) are
    held. A user who appears in any held task keeps all their tasks held."""
    asset, sha, data = _db_asset(domain, out_assets)
    service, policy = _service(domain, asset, sha), (ENVS / domain / 'wiki.md').read_text()
    held = _held_users(domain)
    out = []
    for name in DOMAINS[domain]['task_files']:
        for index, task in enumerate(_tasks(ENVS / domain / name)):
            if task['outputs']:
                continue  # answers told to the customer are not part of the database state
            split = 'validation' if task['user_id'] in held else 'train'
            unit = f'tau-{domain}-user:{task["user_id"]}'
            item = f'tau-{domain}-{name.removesuffix(".py")}-{index}'
            out.append(_case(domain=domain, source=f'tau-{domain}', upstream='sierra-research/tau-bench', item=item,
                             unit=unit, split=split, customer=task['instruction'], actions=task['actions'],
                             policy=policy, service=service, data=data, license_='MIT', key=item, revision='v1'))
    return out


def tau_retail():
    return tau('retail')


def tau_airline():
    return tau('airline')


def apigen(domain, cap=1500, seed=20261031, out_assets=BUILD_V3_ASSETS):
    """APIGen-MT-5k trajectories in one tau-bench domain: the customer's messages become the account of the request;
    the gold actions are the trajectory's own tool calls. Trajectories that touch a user of a held tau-bench task are
    dropped, so held users never appear in training."""
    spec = DOMAINS[domain]
    asset, sha, data = _db_asset(domain, out_assets)
    service, policy = _service(domain, asset, sha), (ENVS / domain / 'wiki.md').read_text()
    held, tools = _held_users(domain), set(spec['reads'] + spec['writes'])
    rows = json.load(open(APIGEN))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out, seen = [], set()
    for index in order:
        conv = rows[index]['conversations']
        calls = [json.loads(turn['value']) for turn in conv if turn['from'] == 'function_call']
        calls = [c for c in calls if isinstance(c, dict) and c.get('name') != 'think']
        if not calls or any(c.get('name') not in tools for c in calls):
            continue  # the other domain's trajectories
        actions = [{'name': c['name'], 'kwargs': (json.loads(c['arguments']) if isinstance(c.get('arguments'), str)
                                                  else c.get('arguments')) or {}} for c in calls]
        if not all(isinstance(a['kwargs'], dict) for a in actions) or not any(a['name'] in spec['writes'] for a in actions):
            continue
        users = _users(domain, actions, data)
        if not users or users & held:
            continue
        messages = [turn['value'] for turn in conv if turn['from'] == 'human']
        customer = 'Messages you sent to customer service, in order:\n' + '\n'.join(f'{i + 1}. {m}' for i, m in enumerate(messages))
        key = text_digest(customer)[:16]
        if key in seen:
            continue
        seen.add(key)
        unit = f'tau-{domain}-user:{sorted(users)[0]}'
        out.append(_case(domain=domain, source=f'apigen-mt-{domain}', upstream='Salesforce/APIGen-MT-5k',
                         item=f'apigen-mt-{index}', unit=unit, split='train', customer=customer, actions=actions,
                         policy=policy, service=service, data=data, license_='CC-BY-NC-4.0', key=key, revision=''))
        if len(out) >= cap:
            break
    return out


def apigen_retail():
    return apigen('retail')


def apigen_airline():
    return apigen('airline')
