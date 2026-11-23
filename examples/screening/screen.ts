/**
 * Screening job applications. Each judgment is a function value: `meets` bound to one requirement with `partial`, and
 * the fit judgment that `fitFor` writes for the role. The screen applies them all at once and keeps the exact
 * bookkeeping in code.
 */
import meets from './meets.nl';
import fitFor from './fit_for.nl';
import type { Application, Check, Screening } from './types.js';

/** Shortlist the applications that meet every requirement and fit the role. */
export async function screen(applications: Application[], role: string, requirements: string[]): Promise<Screening> {
  // One check per requirement: `meets` with its first parameter bound. Then the role's own fit judgment.
  const checks: Check[] = [...requirements.map(requirement => meets.partial(requirement)), await fitFor(role)];
  const results = await Promise.all(applications.map(application => Promise.all(checks.map(check => check(application)))));
  const missing: Record<string, string[]> = {};
  requirements.forEach((requirement, r) => {
    missing[requirement] = applications.filter((_, a) => !results[a]![r]).map(application => application.id);
  });
  const shortlisted = applications.filter((_, a) => results[a]!.every(Boolean)).map(application => application.id);
  return { shortlisted, missing };
}
