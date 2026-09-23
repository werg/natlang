export type Customer = { source_key: string, email: string, name: string };
export type Existing = { id: string, email: string, name: string };
export type Decision = { source_key: string, action: "new" | "merge" | "review", target_email: string, reason: string };
export type Mapping = { customer_id: string, email: string, name: string, order_id: string, order_customer: string, amount: string, unit: string, reason: string };
