---
description: Map two source table shapes into customer and order fields.
args:
  source: string
  customer_columns: string[]
  order_columns: string[]
returns: Mapping
types:
  Mapping: '{ customer_id: string, email: string, name: string, order_id: string, order_customer: string, amount: string, unit: string, reason: string }'
---
Choose actual column names using this exact result shape:
`{customer_id: string, email: string, name: string, order_id: string,
order_customer: string, amount: string, unit: string, reason: string}`.
The values are column names from the supplied customer_columns or order_columns.
`customer_id`, `email`, and `name` refer to customer columns; the other four
refer to order columns. Keep these result field names exactly: do not write
`source_customer_id` or add any other field. `reason` briefly explains ambiguous
choices. The host performs exact cents conversion. A shared display name is
not identity evidence.
Write a `Mapping` record with exactly these fields: `customer_id`, `email`,
`name`, `order_id`, `order_customer`, `amount`, `unit`, and `reason`.
Each of the first seven values is a column name from the appropriate input
list. `reason` briefly explains the mapping choice.
