/** One raw incoming vendor row. External feed payloads are always Untrusted<string>. */
export type RawRow = {
  id: string;
  raw: Untrusted<string>;
};

/** Primitive column types supported by the target ETL schema. */
export type FieldType = 'string' | 'number' | 'currency' | 'date' | 'boolean';

/** Target schema field definition. */
export type SchemaField = {
  name: string;
  type: FieldType;
  required?: boolean;
  aliases?: string[];
};

/** Strict target schema enforced on every output row. */
export type TargetSchema = {
  name: string;
  idField: string;
  amountField: string;
  currencyField: string;
  vendorField: string;
  dateField: string;
  categoryField: string;
  fields: SchemaField[];
};

/**
 * Declarative row-mapping specification synthesized once from sample vendor rows by `buildMapper.nl`
 * and compiled into a zero-token TypeScript transform over all bulk rows.
 */
export type RowMappingSpec = {
  delimiter: 'json' | 'csv' | 'tsv' | 'kv';
  /** Column names in order for delimited (`csv` or `tsv`) rows without inline keys. */
  columns?: string[];
  /** Target field name to ordered candidate source field names/aliases. */
  fieldMap: {
    id: string[];
    amount: string[];
    currency: string[];
    vendor: string[];
    date: string[];
    category: string[];
  };
  /** Multiplier applied to parsed numeric amounts (for example 0.01 when feed reports cents). */
  amountScale: number;
  /** Default ISO-4217 currency code when a row omits currency. */
  defaultCurrency: string;
  /** Date formatting convention used by the vendor feed. */
  dateFormat: 'iso' | 'us' | 'eu' | 'auto';
};

/** Normalized output row satisfying TargetSchema. */
export type CleanRow = {
  id: string;
  amount: number;
  currency: string;
  vendor: string;
  date: string;
  category: string;
  source: 'crisp' | 'semantic';
};

/** Vendor or counterparty entity record for entity resolution. */
export type EntityRecord = {
  id: string;
  name: string;
  country?: string;
  taxId?: string;
  domain?: string;
};

/** Linked entity pair returned by semanticMatch. */
export type EntityPair = {
  left: EntityRecord;
  right: EntityRecord;
  confidence: number;
};

/** Exact conservation and schema invariant report. */
export type ConservationReport = {
  ok: boolean;
  inputRows: number;
  outputRows: number;
  expectedTotal: number | null;
  actualTotal: number;
  problems: string[];
};

/** End-to-end Semantic ETL execution report. */
export type EtlReport = {
  rows: CleanRow[];
  crispRows: number;
  semanticFallbackRows: number;
  crispShare: number;
  conservation: ConservationReport;
  linkedEntities: EntityPair[];
  unmatchedLeft: EntityRecord[];
  unmatchedRight: EntityRecord[];
  mappingAttempts: number;
  mappingSpec: RowMappingSpec;
};
