/**
 * WS2 R2: `search_code(query)` eval builtin and `searchCodebase` discovery tests.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { BUILT_IN_DOCS, searchCodebase } from '../dist/native/runtime.js';
import { scriptedModel } from './support/natlang.mjs';

test('BUILT_IN_DOCS.search_code documents search_code', () => {
  assert.equal(typeof BUILT_IN_DOCS.search_code, 'string');
  assert.match(BUILT_IN_DOCS.search_code, /search_code\(query/);
});

test('searchCodebase matches callable names, signatures, types.ts aliases, descriptions, and service declarations with relevance ranking', () => {
  const codebase = {
    billing: {
      kind: 'namespace',
      name: 'billing',
      source: 'billing',
      codebase: {
        issueRefund: {
          kind: 'natlang',
          id: 'billing/issueRefund.nl',
          name: 'issueRefund',
          source: 'billing/issueRefund.nl',
          revision: 'rev1',
          text: '',
          description: 'Process a customer refund for a disputed invoice.',
          args: { invoice: 'InvoiceRecord', reason: 'string' },
          returns: 'RefundReceipt',
          instructions: 'Check policy and issue refund.',
          types: {
            InvoiceRecord: '{ id: string; amountCents: number }',
            RefundReceipt: '{ refundId: string; approved: boolean }',
          },
          subtype: 'function',
          codebase: {},
        },
        formatCurrency: {
          kind: 'module',
          id: 'billing/formatCurrency.ts',
          name: 'formatCurrency',
          source: 'billing/formatCurrency.ts',
          revision: 'rev2',
          text: 'export function formatCents(cents: number): string { return "$" + (cents / 100).toFixed(2); }',
          types: {},
          imports: [],
          exports: {
            formatCents: {
              kind: 'function',
              args: { cents: 'number' },
              returns: 'string',
              async: false,
              doc: 'Format integer cents into USD currency text.',
            },
          },
          codebase: {},
        },
      },
    },
  };
  const declarations = {
    ledger: `/** Accounting ledger service for settlements and disputes. */\ndeclare namespace ledger {\n  /** Record a settlement entry for an invoice. */\n  function postSettlement(invoiceId: string, amountCents: number): Promise<boolean>;\n}\n`,
  };

  // Exact/prefix name match ranks first.
  const byName = searchCodebase('issueRefund', codebase, { declarations });
  assert.equal(byName[0]?.name, 'billing.issueRefund');
  assert.equal(byName[0]?.kind, 'natlang');
  assert.match(byName[0]?.signature ?? '', /InvoiceRecord.*RefundReceipt/);
  assert.match(byName[0]?.description ?? '', /disputed invoice/i);
  assert.equal(byName[0]?.path, 'billing/issueRefund.nl');

  // Search by return / parameter type finds both the type alias and callables using it.
  const byType = searchCodebase('RefundReceipt', codebase, { declarations });
  assert.ok(byType.some(m => m.name === 'RefundReceipt' && m.kind === 'type'));
  assert.ok(byType.some(m => m.name === 'billing.issueRefund' && m.kind === 'natlang'));
  assert.equal(byType[0]?.name, 'RefundReceipt', 'exact type alias match ranks ahead of signature mention');

  // Case-insensitive tokenized search across doc comments and signatures.
  const byDoc = searchCodebase('USD CURRENCY cents', codebase, { declarations });
  assert.equal(byDoc[0]?.name, 'billing.formatCurrency.formatCents');
  assert.equal(byDoc[0]?.kind, 'function');

  // Service declarations & their methods are searchable by name, param type, or doc comment.
  const byService = searchCodebase('settlement invoiceId', codebase, { declarations });
  assert.equal(byService[0]?.name, 'ledger.postSettlement');
  assert.equal(byService[0]?.kind, 'service');
  assert.match(byService[0]?.signature ?? '', /invoiceId: string/);
});

test('searchCodebase caps results at 20 by default and honors explicit limit', () => {
  const codebase = {};
  for (let i = 0; i < 35; i++) {
    codebase[`helper_${String(i).padStart(2, '0')}`] = {
      kind: 'natlang',
      id: `helper_${i}.nl`,
      name: `helper_${i}`,
      source: `helper_${i}.nl`,
      revision: `r${i}`,
      text: '',
      description: 'Common helper for catalog lookup.',
      args: { query: 'string' },
      returns: 'string',
      instructions: 'Look up catalog item.',
      types: {},
      subtype: 'function',
      codebase: {},
    };
  }
  const defaultMatches = searchCodebase('catalog', codebase);
  assert.equal(defaultMatches.length, 20);
  const limitedMatches = searchCodebase('catalog', codebase, { limit: 5 });
  assert.equal(limitedMatches.length, 5);
});

test('search_code is exposed in eval when the scope has a codebase and omitted when it has none', async () => {
  const withCodebase = loadVirtualNatlang({
    'types.ts': 'export type TicketTag = "billing" | "technical" | "spam";\n',
    'solve.nl': '---\nargs: { text: string }\nreturns: { names: string[]; docs: string }\n---\nDiscover helpers and return their names.\n',
    'solve/classifyTicket.nl': '---\ndescription: Classify a support ticket into a TicketTag.\nargs: { body: string }\nreturns: TicketTag\n---\nPick the matching tag.\n',
    'solve/format.ts': '/** Normalize whitespace in ticket text. */\nexport function normalizeText(raw: string): string { return raw.trim(); }\n',
  }, 'solve.nl');

  const modelWithCode = scriptedModel(() => `
    const found = search_code("TicketTag");
    const docs = read_code("search_code");
    return { names: found.map(m => m.name), docs };
  `);
  const result = await createNatlangRuntime({ model: modelWithCode.driver, calls: false }).run(() => withCodebase('hello'));
  assert.ok(result.names.includes('TicketTag'), `expected TicketTag in ${JSON.stringify(result.names)}`);
  assert.ok(result.names.includes('classifyTicket'), `expected classifyTicket in ${JSON.stringify(result.names)}`);
  assert.match(result.docs, /search_code\(query/);

  // Without a codebase, search_code is not bound in eval so the name stays free for user code / fails as undefined.
  const withoutCodebase = loadVirtualNatlang({
    'plain.nl': '---\nargs: { text: string }\nreturns: string\n---\nReturn whether search_code is defined.\n',
  }, 'plain.nl');
  const modelPlain = scriptedModel(() => 'return typeof search_code;');
  const typeofResult = await createNatlangRuntime({ model: modelPlain.driver, calls: false }).run(() => withoutCodebase('hi'));
  assert.equal(typeofResult, 'undefined');
});
