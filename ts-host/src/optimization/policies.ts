/**
 * The pluggable search policies of the component-search engine: which parent, which kind of experiment (edit or
 * compose) and which components. Each has a crisp default equal to the engine's former behavior and a natural-language
 * side in the authored program (applications/program-improver: improveStep/chooseParent.nl and
 * componentSearchStep/chooseMove.nl, chooseComponents.nl). A setting selects `crisp` (default), `nl` or `shadow`. A
 * natural-language choice is bounded by the crisp verifier in this file; a choice outside the bound throws an error that
 * says what the bound is. The selection math itself is src/gepa.
 */
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { AUTHORED_IMPROVER } from '../improvement/authored-source.js';
import { canonical } from '../adaptation/identity.js';
import { getUpdateGroup } from './vendor/ax-gepa/gepaDependencies.js';
import type { ComponentTarget } from './vendor/ax-gepa/gepaSelection.js';
import type { Candidate } from '../adaptation/types.js';
import type { SearchCandidate } from './types.js';
import { memberOf } from './strategies/gepa.js';
import { wonCases } from '../gepa/index.js';

export type ComponentPolicyName = 'chooseParent' | 'chooseMove' | 'chooseComponents';
export type ComponentPolicySettings = Partial<Record<ComponentPolicyName, PluggableSetting>>;
export type Move = 'edit' | 'compose';
export type ParentChoice = { id: string; quality: number; wonCases: string[]; timesParent: number };
export type MoveFacts = { iteration: number; members: { id: string; changedKeys: string[] }[] };
export type ComponentFacts = { eligible: { key: string; coveredCases: number; failingCases: number; proposals: number; accepts: number }[] };

const NL_FILES: Record<ComponentPolicyName, string> = {
  chooseParent: 'improveStep/chooseParent.nl', chooseMove: 'componentSearchStep/chooseMove.nl', chooseComponents: 'componentSearchStep/chooseComponents.nl' };
const loaded = new Map<string, ReturnType<typeof loadVirtualNatlang>>();
/** The natural-language side of a policy, loaded from the authored program. */
export function policyFunction(name: ComponentPolicyName): (...args: unknown[]) => Promise<any> {
  let callable = loaded.get(name);
  if (!callable) loaded.set(name, callable = loadVirtualNatlang(AUTHORED_IMPROVER, NL_FILES[name]));
  return callable as unknown as (...args: unknown[]) => Promise<any>;
}
/** The settings validated: each names a mode or is absent. */
export function checkPolicySettings(settings: ComponentPolicySettings | undefined): ComponentPolicySettings {
  const known: ComponentPolicyName[] = ['chooseParent', 'chooseMove', 'chooseComponents'];
  for (const key of Object.keys(settings ?? {})) if (!known.includes(key as ComponentPolicyName))
    throw new RangeError(`search policies are ${known.join(', ')}; "${key}" is not one of them`);
  return settings ?? {};
}

const must = (ok: boolean, message: string) => { if (!ok) throw new Error(message); };

/** The facts a parent choice reads: the cases each candidate wins and how often it has been a parent. */
export function parentChoices(population: readonly SearchCandidate[]): ParentChoice[] {
  const members = population.map(memberOf), won = wonCases(members);
  return members.map(member => ({ id: member.id, quality: member.quality, wonCases: won[member.id] ?? [],
    timesParent: population.filter(candidate => candidate.parents.includes(member.id)).length }));
}
/** A parent is a population member. */
export function verifyParent(id: string, population: readonly { id: string }[]): string {
  must(population.some(member => member.id === id), `A parent is the id of a population member; choose one of: ${population.map(member => member.id).join(', ')}.`);
  return id;
}

/** The crisp move: a scheduled composition every fourth experiment of a population-based search. */
export function crispMove(strategy: 'gepa' | 'reflection', iteration: number, members: number): Move {
  return strategy === 'gepa' && iteration % 4 === 3 && members > 1 ? 'compose' : 'edit';
}
/** The component keys each member changed relative to the baseline. */
export function moveFacts(iteration: number, population: readonly SearchCandidate[], baseline: Candidate): MoveFacts {
  return { iteration, members: population.map(member => ({ id: member.id,
    changedKeys: Object.keys(baseline).filter(key => canonical(member.value[key]) !== canonical(baseline[key])).sort() })) };
}
/** A move is `edit` or `compose`, and composing needs two members to merge. */
export function verifyMove(move: string, members: number): Move {
  must(move === 'edit' || move === 'compose', 'A move is `edit` or `compose`.');
  must(move === 'edit' || members > 1, 'The move `compose` merges two population members; the population has one member, so the move is `edit`.');
  return move as Move;
}

/** The keys closed under their dependencies. */
export function closeUnderDependencies(keys: readonly string[], targets: readonly ComponentTarget[]): string[] {
  const closed = new Set<string>();
  for (const key of keys) {
    const target = targets.find(item => item.key === key);
    if (target) for (const member of getUpdateGroup(target, targets)) closed.add(member.key);
  }
  return [...closed];
}
/** Chosen keys are eligible, which the host then closes under their dependencies. */
export function verifyComponents(keys: readonly string[], eligible: readonly string[]): string[] {
  must(keys.length > 0, 'Choose at least one component key.');
  const outside = keys.filter(key => !eligible.includes(key));
  must(!outside.length, `Component keys are among the eligible ones (${eligible.join(', ')}); these are not: ${outside.join(', ')}.`);
  return [...keys];
}

/** The facts a component choice reads: per eligible key, the parent's training cases that cover it and the edits tried. */
export function componentFactsOf(keys: readonly string[], parent: SearchCandidate, selector: Readonly<Record<string, { proposals: number; accepts: number }>>): ComponentFacts {
  return { eligible: keys.map(key => {
    const covering = parent.train.results.filter(result => [...result.coverage].includes(key));
    return { key, coveredCases: covering.length, failingCases: covering.filter(result => (result.quality ?? 0) < 1).length,
      proposals: selector[key]?.proposals ?? 0, accepts: selector[key]?.accepts ?? 0 };
  }) };
}

/** One pluggable decision: `crisp` and `nl` are the two implementations of the same part, served per the setting. */
export function decide<A extends unknown[], R>(name: ComponentPolicyName, setting: PluggableSetting, crisp: (...args: A) => R | Promise<R>, nl: (...args: A) => R | Promise<R>) {
  return pluggable<A, R>({ crisp, nl }, setting, { name, default: 'crisp', serve: 'crisp' });
}
