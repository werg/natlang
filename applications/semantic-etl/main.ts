/** `natlang run applications/semantic-etl -- [INPUT.json]` (or `--rows JSON --schema JSON`, or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { runSemanticEtl, type EtlOptions } from './etl.js';
import type { RawRow, TargetSchema } from './types.js';

const DEFAULT_INVOICE_SCHEMA: TargetSchema = {
  name: 'vendor_invoices_v1',
  idField: 'id',
  amountField: 'amount',
  currencyField: 'currency',
  vendorField: 'vendor',
  dateField: 'date',
  categoryField: 'category',
  fields: [
    { name: 'id', type: 'string', required: true },
    { name: 'amount', type: 'currency', required: true },
    { name: 'currency', type: 'string', required: true },
    { name: 'vendor', type: 'string', required: true },
    { name: 'date', type: 'date', required: true },
    { name: 'category', type: 'string', required: true },
  ],
};

type CliPayload = {
  rows: RawRow[];
  schema?: TargetSchema;
  expectedTotal?: number;
  expectedSampleTotal?: number;
  sampleSize?: number;
  entityLink?: EtlOptions['entityLink'];
  options?: Omit<EtlOptions, 'block'>;
};

function readJsonInput(context: TargetContext, filePath?: string): CliPayload {
  const path = filePath ?? context.args[0];
  const raw = path
    ? readFileSync(resolve(context.workspace, path), 'utf8')
    : readFileSync(0, 'utf8');
  return JSON.parse(raw) as CliPayload;
}

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as {
    file?: string;
    rows?: RawRow[];
    schema?: TargetSchema;
    expectedTotal?: number;
    entityLink?: EtlOptions['entityLink'];
  };

  let rows: RawRow[];
  let schema: TargetSchema;
  let options: EtlOptions;

  if (args.rows) {
    rows = args.rows;
    schema = args.schema ?? DEFAULT_INVOICE_SCHEMA;
    options = {
      ...(args.expectedTotal !== undefined ? { expectedTotal: args.expectedTotal } : {}),
      ...(args.entityLink ? { entityLink: args.entityLink } : {}),
    };
  } else {
    const payload = readJsonInput(context, args.file);
    rows = payload.rows ?? [];
    schema = payload.schema ?? DEFAULT_INVOICE_SCHEMA;
    options = {
      ...(payload.options ?? {}),
      ...(payload.expectedTotal !== undefined ? { expectedTotal: payload.expectedTotal } : {}),
      ...(payload.expectedSampleTotal !== undefined ? { expectedSampleTotal: payload.expectedSampleTotal } : {}),
      ...(payload.sampleSize !== undefined ? { sampleSize: payload.sampleSize } : {}),
      ...(payload.entityLink ? { entityLink: payload.entityLink } : {}),
    };
  }

  const report = await context.runtime.run(() => runSemanticEtl(rows, schema, options));
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return 0;
}
