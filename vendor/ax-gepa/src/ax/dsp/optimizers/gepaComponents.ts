import type { AxGenOut, AxProgrammable } from '../types.js';

export type AxGEPAComponentTarget = {
  id: string;
  kind: string;
  current: string;
  description?: string;
  constraints?: string;
  traceId?: string;
  dependsOn?: readonly string[];
  preserve?: readonly string[];
  maxLength?: number;
  format?: string;
  validate?: (value: string) => true | string;
};

export function getGEPAOptimizationTargets<IN, OUT extends AxGenOut>(
  program: Readonly<AxProgrammable<IN, OUT>>
): AxGEPAComponentTarget[] {
  const fn = (program as any).getOptimizableComponents;
  if (typeof fn !== 'function') return [];

  const components = fn.call(program) as readonly {
    key: string;
    kind: string;
    current: string;
    description?: string;
    constraints?: string;
    traceId?: string;
    dependsOn?: readonly string[];
    preserve?: readonly string[];
    maxLength?: number;
    format?: string;
    validate?: (value: string) => true | string;
  }[];

  const out: AxGEPAComponentTarget[] = [];
  const seen = new Set<string>();
  for (const c of components) {
    if (!c?.key || seen.has(c.key)) continue;
    if (typeof c.current !== 'string') continue;
    seen.add(c.key);
    out.push({
      id: c.key,
      kind: c.kind,
      current: c.current,
      description: c.description,
      constraints: c.constraints,
      traceId: c.traceId,
      dependsOn: c.dependsOn,
      preserve: c.preserve,
      maxLength: c.maxLength,
      format: c.format,
      validate: c.validate,
    });
  }
  return out;
}

export function applyGEPAComponentConfig<IN, OUT extends AxGenOut>(
  program: Readonly<AxProgrammable<IN, OUT>>,
  cfg: Readonly<Record<string, string>>
): void {
  const fn = (program as any).applyOptimizedComponents;
  if (typeof fn === 'function') {
    fn.call(program, cfg);
  }
}
