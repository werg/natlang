/** Shared by live inference and corpus replay so capability instructions do not drift. */
export function applicationCapabilityPrompt(capabilities: { allowModules: boolean; allowNetwork: boolean },
  availableDependencies: readonly string[] = []): string {
  const available = [...new Set(availableDependencies)].sort();
  return (capabilities.allowModules ?
    '\nApplication packages are resolved from the configured app workspace. In eval, use await installPackages(["package@version"]) to update package.json and package-lock.json. Imports never install packages implicitly. Static imports and await import("package") are supported; imported bindings are local to the current eval, so re-import in later calls. Relative imports resolve from the application root. Node built-ins can be imported with node: specifiers. Package installation has external effects that are not rolled back on eval failure.\n' +
    (available.length ? `Importable application dependencies from package.json:\n${available.map(name => `- ${JSON.stringify(name)}`).join('\n')}\n` :
      'No declared application dependencies are currently installed. If package.json declares dependencies, use await installPackages([]) before importing them.\n') : '') +
    (capabilities.allowNetwork ? '\nNetwork access is available through fetch; consume responses into portable values before returning. HTTP response handles are local to the current eval.\n' : '');
}
