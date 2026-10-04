#!/usr/bin/env node
/** Original, deterministic operational time-series episodes with host-graded assignment outputs. */
import { createHash } from 'node:crypto';
import { mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const metric = { schema: 'natlang.skill-graded/1', kind: 'assignment-accuracy' };
const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const json = value => JSON.stringify(value);

const families = [
  { id: 'occupancy-calendar', title: 'Room occupancy and calendar effects', rule:
    'For each observation, expected occupancy is the baseline for its calendar class plus the listed scheduled-event adjustment. Label it calendar-explained exactly when observed occupancy equals that value; otherwise label it residual.', templates: [
    { id: 'weekday-weekend-baseline', baseline: { weekday: 20, weekend: 34 }, event: 0, story: 'A community arts hall has different normal use on weekdays and weekends.', rows: [['mon', 'weekday', 20], ['sat', 'weekend', 34], ['tue', 'weekday', 23]] },
    { id: 'evening-event-adjustment', baseline: { weekday: 18, weekend: 30 }, event: 12, story: 'An evening lecture adds a known number of attendees to the ordinary room count.', rows: [['wed-early', 'weekday', 18, false], ['wed-event', 'weekday', 30, true], ['thu-event', 'weekday', 27, true]] },
    { id: 'holiday-calendar', baseline: { weekday: 22, holiday: 8 }, event: 6, story: 'A public holiday changes staffing and an announced tour adds a fixed group.', rows: [['fri', 'weekday', 22, false], ['holiday', 'holiday', 14, true], ['holiday-late', 'holiday', 15, false]] },
    { id: 'shifted-baseline', baseline: { weekday: 15, weekend: 28, festival: 40 }, event: 9, story: 'A temporary festival schedule has its own baseline and an extra scheduled performance.', rows: [['weekday', 'weekday', 15, false], ['festival', 'festival', 40, false], ['show', 'festival', 49, true], ['weekend', 'weekend', 31, false]] },
  ], make(t) { return { family: this.id, title: this.title, story: t.story, baseline: t.baseline, scheduledEventAdjustment: t.event,
    rule: `${this.rule} Baselines: ${json(t.baseline)}. Scheduled event adjustment: +${t.event}.`,
    observations: t.rows.map(([slot, calendarClass, observed, scheduled = false]) => ({ slot, calendarClass, scheduledEvent: scheduled, observed })) };
  }, answer(x) { return Object.fromEntries(x.observations.map(row => [row.slot,
    row.observed === x.baseline[row.calendarClass] + (row.scheduledEvent ? x.scheduledEventAdjustment : 0) ? 'calendar-explained' : 'residual'])); } },

  { id: 'maintenance-incidents', title: 'Planned maintenance and unplanned incidents', rule:
    'A reading below the minimum is planned-maintenance only when it falls inside a listed work-order interval for the same asset. A below-minimum reading outside such an interval is an incident; a reading at or above the minimum is normal.', templates: [
    { id: 'single-window', minimum: 50, story: 'A pump is isolated during a scheduled seal replacement.', workOrders: [{ asset: 'pump-A', start: 2, end: 3 }], rows: [['a1', 'pump-A', 1, 80], ['a2', 'pump-A', 2, 0], ['a3', 'pump-A', 3, 0], ['a4', 'pump-A', 4, 20]] },
    { id: 'two-assets', minimum: 40, story: 'A chiller and a fan have separate maintenance calendars.', workOrders: [{ asset: 'chiller', start: 1, end: 1 }, { asset: 'fan', start: 3, end: 4 }], rows: [['c0', 'chiller', 0, 70], ['c1', 'chiller', 1, 0], ['f3', 'fan', 3, 0], ['f5', 'fan', 5, 15]] },
    { id: 'inspection-gap', minimum: 60, story: 'A conveyor inspection is planned for one shift; adjacent shifts remain in service.', workOrders: [{ asset: 'belt-2', start: 4, end: 4 }], rows: [['b3', 'belt-2', 3, 74], ['b4', 'belt-2', 4, 0], ['b5', 'belt-2', 5, 52], ['b6', 'belt-2', 6, 75]] },
    { id: 'cross-midnight-work', minimum: 30, story: 'A boiler service interval crosses midnight and is recorded on one continuous shift index.', workOrders: [{ asset: 'boiler', start: 6, end: 7 }], rows: [['d5', 'boiler', 5, 55], ['d6', 'boiler', 6, 0], ['d7', 'boiler', 7, 0], ['d8', 'boiler', 8, 12]] },
  ], make(t) { return { family: this.id, title: this.title, story: t.story, minimumOperatingReading: t.minimum, workOrders: t.workOrders,
    rule: `${this.rule} Minimum operating reading: ${t.minimum}. Work orders: ${json(t.workOrders)}.`,
    observations: t.rows.map(([slot, asset, shift, reading]) => ({ slot, asset, shift, reading })) };
  }, answer(x) { return Object.fromEntries(x.observations.map(row => { const low = row.reading < x.minimumOperatingReading; const planned = x.workOrders.some(order => order.asset === row.asset && row.shift >= order.start && row.shift <= order.end); return [row.slot, low ? (planned ? 'planned-maintenance' : 'incident') : 'normal']; })); } },

  { id: 'meter-resets', title: 'Meter resets and consumption', rule:
    'Classify the first cumulative reading as initial-reading. For each later reading, label it reset when its timestamp appears in resetLog. Otherwise classify it as ordinary-consumption when it is at least the preceding reading, and as unlogged-drop when it is lower. Resets are not negative consumption.', templates: [
    { id: 'explicit-midseries-reset', story: 'A water meter was replaced after a technician recorded a register reset.', resetLog: ['t3'], rows: [['t1', 100], ['t2', 118], ['t3', 4], ['t4', 13]] },
    { id: 'no-reset-drop', story: 'A cold-storage meter has no maintenance reset on file during the observed week.', resetLog: [], rows: [['m1', 40], ['m2', 45], ['m3', 42], ['m4', 50]] },
    { id: 'two-resets', story: 'A test meter is deliberately zeroed after each calibration run.', resetLog: ['r2', 'r4'], rows: [['r1', 8], ['r2', 0], ['r3', 7], ['r4', 0], ['r5', 6]] },
    { id: 'late-reset-entry', story: 'The plant log records a reset at the third sample; a later unexplained decrease has no corresponding work record.', resetLog: ['p3'], rows: [['p1', 210], ['p2', 224], ['p3', 2], ['p4', 16], ['p5', 11]] },
  ], make(t) { return { family: this.id, title: this.title, story: t.story,
    rule: this.rule, resetLog: t.resetLog, observations: t.rows.map(([slot, cumulative]) => ({ slot, cumulative })) };
  }, answer(x) { const resets = new Set(x.resetLog); let previous; return Object.fromEntries(x.observations.map(row => {
    const label = previous === undefined ? 'initial-reading' : resets.has(row.slot) ? 'reset' : row.cumulative >= previous ? 'ordinary-consumption' : 'unlogged-drop';
    previous = row.cumulative; return [row.slot, label]; })); } },

  { id: 'sensor-calibration', title: 'Sensor units and calibration changes', rule:
    'For each reading, multiply rawValue by the calibration factor active at that time. Label the corrected reading in-range when it is within the inclusive operating range; otherwise label it out-of-range.', templates: [
    { id: 'factor-change', low: 18, high: 22, story: 'A temperature probe changed scale after a documented recalibration.', rows: [['s1', 1, 20], ['s2', 2, 10], ['s3', 2, 12], ['s4', 2, 14]] },
    { id: 'unit-conversion', low: 90, high: 110, story: 'A flow sensor reports in half-scale units after its controller configuration changed.', rows: [['u1', 1, 100], ['u2', 2, 50], ['u3', 2, 56], ['u4', 2, 40]] },
    { id: 'offset-band', low: 5, high: 8, story: 'A conductivity sensor factor changed while the allowed corrected band stayed fixed.', rows: [['o1', 1, 6], ['o2', 1, 9], ['o3', 0.5, 14], ['o4', 0.5, 12]] },
    { id: 'narrow-operating-band', low: 48, high: 52, story: 'A pressure sensor was calibrated to a tighter operating band before the final readings.', rows: [['n1', 1, 50], ['n2', 0.5, 100], ['n3', 0.5, 108], ['n4', 0.5, 88]] },
  ], make(t) { return { family: this.id, title: this.title, story: t.story, operatingRange: [t.low, t.high],
    rule: `${this.rule} Inclusive operating range: ${t.low} through ${t.high}.`,
    observations: t.rows.map(([slot, calibrationFactor, rawValue]) => ({ slot, calibrationFactor, rawValue })) };
  }, answer(x) { const [low, high] = x.operatingRange;
    return Object.fromEntries(x.observations.map(row => { const corrected = row.rawValue * row.calibrationFactor; return [row.slot, corrected >= low && corrected <= high ? 'in-range' : 'out-of-range']; })); } },

  { id: 'delayed-intervention', title: 'Delayed intervention effects', rule:
    'The expected reading starts at baseline. Each intervention adds its stated delta beginning exactly `delay` sample steps after its startIndex, and remains active thereafter. Label a sample consistent only when observed equals the resulting expected reading.', templates: [
    { id: 'two-step-response', baseline: 10, interventions: [{ startIndex: 0, delay: 2, delta: 5 }], values: [10, 10, 15, 15, 14], story: 'A ventilation change has a two-sample response lag.' },
    { id: 'immediate-response', baseline: 30, interventions: [{ startIndex: 1, delay: 0, delta: -4 }], values: [30, 26, 26, 26, 25], story: 'A valve adjustment affects flow immediately after its logged sample.' },
    { id: 'overlapping-effects', baseline: 5, interventions: [{ startIndex: 0, delay: 1, delta: 3 }, { startIndex: 2, delay: 2, delta: -2 }], values: [5, 8, 8, 8, 6], story: 'A treatment and a later offset have independent delayed effects.' },
    { id: 'three-step-response', baseline: 100, interventions: [{ startIndex: 0, delay: 3, delta: -10 }], values: [100, 100, 100, 90, 90, 88], story: 'A filter replacement takes three sample intervals to reach its steady response.' },
  ], make(t) { return { family: this.id, title: this.title, story: t.story,
    rule: this.rule, baseline: t.baseline, interventions: t.interventions,
    observations: t.values.map((observed, index) => ({ slot: `sample-${index}`, index, observed })) };
  }, answer(x) { return Object.fromEntries(x.observations.map(row => { const predicted = x.baseline + x.interventions.reduce((sum, event) => row.index >= event.startIndex + event.delay ? sum + event.delta : sum, 0);
    return [row.slot, row.observed === predicted ? 'model-consistent' : 'deviation']; })); } },

  { id: 'inventory-backlog', title: 'Inventory backlog and demand', rule:
    'Available stock equals openingStock plus receipts. Required units equal currentDemand plus backlogIn. Expected shipment is the smaller of available stock and required units. Label process-shortfall if observed shipped is below expected shipment; otherwise label supply-constrained when required units exceed available stock, and stock-sufficient when they do not. Do not infer demand from shipments alone.', templates: [
    { id: 'stockout-after-spike', story: 'A warehouse receives a one-day demand spike before its replenishment truck arrives.', rows: [['d1', 12, 0, 8, 0, 8], ['d2', 4, 0, 9, 0, 4], ['d3', 0, 10, 5, 0, 5]] },
    { id: 'backlog-carry', story: 'Orders left unfilled yesterday remain part of the fulfillment obligation today.', rows: [['b1', 5, 0, 4, 0, 4], ['b2', 1, 0, 3, 1, 1], ['b3', 0, 8, 2, 2, 2]] },
    { id: 'steady-replenishment', story: 'A clinic replenishes supplies every period while urgent and routine demand vary.', rows: [['c1', 8, 2, 5, 0, 5], ['c2', 5, 3, 7, 0, 7], ['c3', 1, 10, 6, 0, 6]] },
    { id: 'demand-vs-supply', story: 'A parts desk records low shipments both when demand is low and when stock is unavailable.', rows: [['p1', 0, 3, 2, 0, 2], ['p2', 1, 0, 5, 0, 1], ['p3', 0, 8, 3, 0, 3]] },
  ], make(t) { return { family: this.id, title: this.title, story: t.story,
    rule: this.rule, observations: t.rows.map(([slot, openingStock, receipts, currentDemand, backlogIn, shipped]) => ({ slot, openingStock, receipts, currentDemand, backlogIn, shipped })) };
  }, answer(x) { return Object.fromEntries(x.observations.map(row => { const available = row.openingStock + row.receipts, required = row.currentDemand + row.backlogIn;
    const expectedShipment = Math.min(available, required);
    return [row.slot, row.shipped < expectedShipment ? 'process-shortfall' : required > available ? 'supply-constrained' : 'stock-sufficient']; })); } },
];

const variants = {
  empty: { kind: 'empty', skills: {} },
  'general-guide': { kind: 'existing', skills: { 'series-reading': { 'SKILL.md': '---\nname: series-reading\ndescription: Write concise reports about sequences of measurements.\n---\nWhen reading a sequence, keep its timestamps in order. Separate recorded events from measured values. Apply the stated rule to each point and return only the requested fields.\n' } } },
  'metadata-tuned': { kind: 'existing', skills: { 'series-reading': { 'SKILL.md': '---\nname: series-reading\ndescription: Interpret operational time series using calendar context, work orders, reset logs, calibration records, intervention delays, and inventory flow rules.\n---\nWhen reading a sequence, keep its timestamps in order. Separate recorded events from measured values. Apply the stated rule to each point and return only the requested fields.\n' } } },
};

const target = { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'natlang.timeseries-task/1', id: 'operational-timeseries-classification-v1' },
  files: { 'solve.nl': '---\nargs: { scenario: string }\nreturns: string\n---\nRead the operational story, explicit rules, event records, and ordered time-series observations in the scenario JSON. For every observation, determine the category required by the stated rule. Return a JSON object mapping each observation slot to its category, with no extra keys or prose.\n' } };

function scenarioData(family, template) {
  return family.make(template);
}
function buildTimeSeriesEpisodes() {
  const episodes = [];
  for (const family of families) {
    for (const [templateIndex, template] of family.templates.entries()) {
      const scenario = scenarioData(family, template);
      const group = `timeseries-template/${family.id}/${template.id}`;
      const row = { id: `ts-${digest(group).slice(0, 20)}`, group, args: [json(scenario)],
        expected: { kind: 'assignment', value: family.answer(scenario) } };
      for (const [variant, library] of Object.entries(variants)) {
        const id = `timeseries-${family.id}-${variant}`;
        const episode = episodes.find(item => item.id === id) ?? {
          version: 'natlang.skill-episode/1', id, family: `timeseries:${family.id}`, split: 'train',
          source_groups: family.templates.map(item => `timeseries-template/${family.id}/${item.id}`),
          license: 'project-generated', target, library: structuredClone(library), support: { cases: [] }, query: { cases: [] },
          operations: ['create', 'revise'], limits: { maxSteps: 6 },
          provenance: { generator: 'natlang.semantic-timeseries-episodes/1', variant,
            template_holdout: { support: family.templates.slice(0, 2).map(item => item.id), query: family.templates.slice(2).map(item => item.id) },
            metric, grading: 'per-observation assignment accuracy; no runtime or subjective score' },
        };
        if (!episodes.includes(episode)) episodes.push(episode);
        (templateIndex < 2 ? episode.support.cases : episode.query.cases).push(structuredClone(row));
      }
    }
  }
  return episodes;
}

export { buildTimeSeriesEpisodes, families, metric, variants };

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const at = process.argv.indexOf('--out'), argument = at >= 0 ? process.argv[at + 1] : undefined;
  if (!argument || argument.startsWith('--')) throw Error('Usage: build-timeseries-episodes.mjs --out DIR');
  const out = resolve(argument);
  const episodes = buildTimeSeriesEpisodes();
  mkdirSync(out, { recursive: true });
  const body = episodes.map(episode => JSON.stringify(episode)).join('\n') + '\n';
  const output = join(out, 'timeseries-episodes.jsonl');
  const fd = openSync(output, 'wx');
  try { writeFileSync(fd, body); } finally { closeSync(fd); }
  const manifest = { schema: 'natlang.semantic-timeseries-episodes/1', episodes: episodes.length,
    families: families.length, cases: episodes.reduce((sum, row) => sum + row.support.cases.length + row.query.cases.length, 0),
    variants: Object.keys(variants), metric, model_calls: 0, sha256: digest(body) };
  const manifestFd = openSync(join(out, 'timeseries-episodes.manifest.json'), 'wx');
  try { writeFileSync(manifestFd, JSON.stringify(manifest, null, 2) + '\n'); } finally { closeSync(manifestFd); }
  process.stdout.write(`${JSON.stringify(manifest)}\n`);
}
