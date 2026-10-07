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
