/** Per-run host inputs for staged applications. Factories avoid stale filesystem views. */
export type ApplicationInputs = Record<string, unknown> | (() => Record<string, unknown>);

export function resolveApplicationInputs(inputs?: ApplicationInputs): Record<string, unknown> | undefined {
  return typeof inputs === 'function' ? inputs() : inputs;
}
