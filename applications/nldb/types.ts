/** catalog.json at the database root: the schema of every table. */
export type Catalog = {
  /** Increases by one with every committed schema change. */
  version: number,
  /**
   * Each table by name. Its rows live in tables/<table>/<page>.jsonl, one JSON object per line ({"_id": 7, ...one
   * field per column, null when absent}), pages 0, 1, 2, … of at most 256 rows ordered by _id. An index is
   * indexes/<table>/<column>.json: a JSON array of [value, _id] pairs sorted by value, then _id.
   */
  tables: Record<string, Table>,
};

export type Table = {
  /** What one row stands for, in a sentence. */
  description: string,
  columns: Column[],
  /** The columns whose values identify a row; ["_id"] when there is no natural key. */
  key: string[],
  /** The columns that have an index file. */
  indexes: string[],
  /** How many page files the table has. */
  pages: number,
  rows: number,
  /** The _id the next inserted row gets; every row has a unique integer _id. */
  nextId: number,
};

export type Column = {
  name: string,
  /** date is YYYY-MM-DD, timestamp is ISO 8601, neuralese is the ID of a stored neuralese block. */
  type: 'integer' | 'real' | 'text' | 'boolean' | 'date' | 'timestamp' | 'neuralese',
  /** What the value means, with its unit when it has one. */
  description: string,
  nullable: boolean,
  unique: boolean,
  /** "table.column" this column refers to (a foreign key), or null. */
  references: string | null,
  /** A condition every value must meet, in words ("at least 0"), or null. */
  check: string | null,
};

/** What a request is. */
export type Kind = 'question' | 'change' | 'schema' | 'unclear';

/** What the database answers a request with. A request that failed changed nothing. */
export type Outcome = { kind: 'question', answer: Answer } | { kind: 'change' | 'schema', report: Report } |
  { kind: 'unclear', clarification: string } | { kind: Kind, error: string };

/** A stored value. Dates and timestamps are text; a neuralese column holds a block ID. */
export type Value = string | number | boolean | null;
/** A table's row as stored: _id and a field per column, by plain column name. */
export type Stored = Record<string, Value>;
/** A row between operators: values by qualified name, alias.column (alias._id included), or by output name. */
export type Row = Record<string, Value>;

/** A condition on meaning, judged per distinct value: a row passes when its column's value meets criterion. */
export type Semantic = { column: string, criterion: string };
/** An output column: its name and the SQL expression computing it ("c.name", "count(*)", "sum(o.total)"). */
export type Output = { name: string, expression: string };

/**
 * One relational operator. A step reads a table (scan, lookup) or the rows of earlier steps, named by position
 * (input, left, right; 0 is the plan's first step). where and on are SQL conditions over qualified names
 * ("c.city = 'Lisbon' AND c.signed_up >= '2026-09-01'"); a null where keeps every row.
 * - scan: every row of table, read page by page, that meets where.
 * - lookup: the rows of table whose column holds one of values, or lies in range (a null bound is open), found
 *   through the column's index; then where.
 * - filter: the rows of input that meet where and every semantic condition.
 * - join: left's rows combined with right's on on: inner pairs, left (unmatched left rows kept with nulls), semi
 *   (left rows with a match), anti (left rows without).
 * - aggregate: one row per group of input by the groupBy expressions, with each aggregate.
 * - project: input's rows with the columns computed; distinct drops repeated rows.
 * - sort: input's rows ordered by the expressions, then offset rows skipped and limit kept (all when null).
 */
export type Step = { op: 'scan', table: string, alias: string, where: string | null } |
  { op: 'lookup', table: string, alias: string, column: string, values: Value[] | null, range: { low: Value, high: Value } | null, where: string | null } |
  { op: 'filter', input: number, where: string | null, semantic: Semantic[] } |
  { op: 'join', left: number, right: number, on: string, kind: 'inner' | 'left' | 'semi' | 'anti', method: 'hash' | 'nested-loop' } |
  { op: 'aggregate', input: number, groupBy: string[], aggregates: Output[] } |
  { op: 'project', input: number, columns: Output[], distinct: boolean } |
  { op: 'sort', input: number, by: { expression: string, descending: boolean }[], limit: number | null, offset: number };

/** A query plan: its steps in order. The last step's rows are the result. */
export type Plan = { steps: Step[] };

/**
 * One change of a transaction.
 * - insert: new rows of table, by plain column name without _id: rows as given, or the rows from's plan yields
 *   (alias prefixes dropped).
 * - update: target yields the rows to change, read under alias (alias._id and alias.column); set gives each changed
 *   column's new value as an expression over target's row.
 * - delete: target yields the rows to delete, read under alias.
 */
export type Change = { op: 'insert', table: string, rows: Stored[], from: Plan | null } |
  { op: 'update', table: string, alias: string, set: Output[], target: Plan } |
  { op: 'delete', table: string, alias: string, target: Plan };

/** A request as the database reads it, with the SQL it amounts to. */
export type Statement = { kind: 'question', sql: string[], plan: Plan, columns: string[], explanation: string, assumptions: string[] } |
  { kind: 'change', sql: string[], changes: Change[], assumptions: string[] };

/** A row before and after a change, as stored: before is null for an insert, after for a delete. */
export type RowChange = { before: Stored | null, after: Stored | null };

/** What running a statement gave: a question's rows (values in the statement's column order), a change's counts per table. */
export type Execution = { rows: Value[][], changes: Record<string, { inserted: number, updated: number, deleted: number }> };

/**
 * One schema change, in DDL's terms. An added column's default is every stored row's value for it (null when none);
 * constrain column gives the column's constraints after the change.
 */
export type SchemaChange = { op: 'create table', table: string, description: string, columns: Column[], key: string[], indexes: string[] } |
  { op: 'drop table', table: string } |
  { op: 'rename table', table: string, to: string } |
  { op: 'add column', table: string, column: Column, default: Value } |
  { op: 'drop column', table: string, column: string } |
  { op: 'rename column', table: string, column: string, to: string } |
  { op: 'create index', table: string, column: string } |
  { op: 'drop index', table: string, column: string } |
  { op: 'constrain column', table: string, column: string, nullable: boolean, unique: boolean, references: string | null, check: string | null };

/** A schema request designed: the changes in order, the DDL they amount to, and what was assumed. */
export type Design = { changes: SchemaChange[], ddl: string[], assumptions: string[] };

/** The answer to a question. */
export type Answer = {
  columns: string[],
  rows: (string | number | boolean | null)[][],
  /** How the answer was found: what was read, matched, joined, grouped and computed, in two or three sentences. */
  explanation: string,
  /** What was assumed where the question allowed more than one reading; empty when nothing was. */
  assumptions: string[],
};

/** What a committed change did. */
export type Report = {
  /** The request read as relational statements, in order, e.g. "UPDATE accounts SET balance = balance - 50 WHERE owner = 'Ana'". */
  statements: string[],
  /** Rows inserted, updated and deleted per table. */
  changes: Record<string, { inserted: number, updated: number, deleted: number }>,
  assumptions: string[],
  summary: string,
};

/** A request compiled to SQLite. */
export type SqlPlan = {
  kind: Kind,
  /** SQLite statements to run in order as one transaction; a question is one SELECT. */
  statements: string[],
  /**
   * Conditions on meaning, written in the statements as meets(column, 'criterion'), with the table and column each
   * applies to; empty when every condition is exact.
   */
  semantic: { criterion: string, table: string, column: string }[],
  /** For an unclear request, the one question that would settle it; otherwise empty. */
  clarification: string,
  explanation: string,
  assumptions: string[],
};
