import { fingerprint } from '../adaptation/identity.js';
import type { FixtureEntryBinding } from './types.js';

/** A host-only facade along one declared export path; original functions and their live arguments execute unchanged. */
export function fixtureEntryFacade(exports: Record<string, unknown>, binding: FixtureEntryBinding,
  observe: () => void): Record<string, unknown> {
  const wrap = (owner: unknown, path: readonly string[]): unknown => {
    if (!owner || (typeof owner !== 'object' && typeof owner !== 'function'))
      throw new Error(`fixture entry export path ${binding.export_path.join('.')} is absent`);
    const [name, ...rest] = path;
    const value = (owner as Record<string, unknown>)[name!];
    const replacement = rest.length ? wrap(value, rest) : (...args: unknown[]) => {
      if (typeof value !== 'function') throw new Error('fixture entry export is not callable');
      if (args.length > binding.input_arguments.length) throw new Error('fixture entry received undeclared arguments');
      for (const [index, argument] of binding.input_arguments.entries()) {
        if (binding.service_inputs.includes(argument)) {
          if (args[index] === null || !['object', 'function'].includes(typeof args[index]))
            throw new Error(`fixture entry ${argument} must be constructed by the host fixture`);
        } else if (fingerprint(args[index]) !== fingerprint(binding.inputs[argument]))
          throw new Error(`fixture entry argument ${argument} differs from the program IR input`);
      }
      observe();
      return Reflect.apply(value, owner, args);
    };
    const facade: object = typeof owner === 'function' ? (...args: unknown[]) => Reflect.apply(owner, undefined, args) : {};
    for (const key of Reflect.ownKeys(owner)) {
      if (['name', 'length', 'prototype', 'caller', 'arguments'].includes(String(key))) continue;
      const descriptor = Object.getOwnPropertyDescriptor(owner, key)!;
      Object.defineProperty(facade, key, key === name
        ? { value: replacement, writable: false, enumerable: descriptor.enumerable, configurable: true }
        : { ...descriptor, configurable: true });
    }
    if (!Object.hasOwn(owner, name!)) throw new Error(`fixture entry export ${name} is absent`);
    return facade;
  };
  return wrap(exports, binding.export_path) as Record<string, unknown>;
}
