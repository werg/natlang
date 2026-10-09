/** The registry the Harness reads: the application's registry, with the built-in task names resolved to the port's kinds. */
import type { AnyTask, RegistryReader, RegistrySnapshot } from '../vendor/durable/src/harness/types.ts';

/** A registry reader whose snapshots resolve the built-in task names to the natural-language kinds. */
export function substituteTasks(registry: RegistryReader, tasks: readonly AnyTask[]): RegistryReader {
  const byName = new Map(tasks.map(task => [task.definition.name, task]));
  const wrap = (snapshot: RegistrySnapshot): RegistrySnapshot => ({
    installed: () => snapshot.installed(),
    extension: name => snapshot.extension(name),
    tools: () => snapshot.tools(),
    sections: () => snapshot.sections(),
    tasks: () => snapshot.tasks().map(task => byName.get(task.definition.name) ?? task),
    // Only a task the registry has is substituted: a registry without a built-in task stays without it.
    task: name => { const own = snapshot.task(name); return own && (byName.get(name) ?? own); },
  });
  let source: RegistrySnapshot | undefined, wrapped: RegistrySnapshot | undefined;
  return {
    snapshot() {
      const current = registry.snapshot();
      if (current !== source) { source = current; wrapped = wrap(current); }
      return wrapped!;
    },
    subscribe: listener => registry.subscribe(listener),
  };
}

