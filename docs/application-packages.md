# Application packages and network access

A Node-backed natlang application owns a normal `package.json` and
`package-lock.json`. Its dependencies belong to the application, not an individual
lambda or eval session. Both handwritten `.ts` functions and generated eval code
resolve packages through that application workspace.

```ts
const host = new NativeNatlangHost({ workspace: '/absolute/path/to/app' });
// Explicit preparation: npm ci when a lockfile exists, otherwise npm install.
await host.environment.packages!.prepareDependencies();
await host.run({ source: { kind: 'file', path: '/absolute/path/to/app/main.ts' } });
host.close();
```

An explicit workspace requires an existing package.json. Construction does not install anything.
The `natlang` CLI finds the nearest `package.json` above a program file (or the
current directory for an anonymous instruction); `--workspace /absolute/app-root`
selects one explicitly. The Node eval environment also finds the current
project automatically; import syntax is not controlled by a workspace mode.
If no project manifest exists, an attempted package import reports that a
`package.json` is required.
`prepareDependencies()` respects the existing lockfile. `installPackages(specs)`
runs npm install and updates package.json and package-lock.json. Concurrent installs
in the same workspace are serialized within the host process. npm lifecycle scripts
are enabled. Package specifiers can include versions, tags, Git URLs, and local
packages accepted by npm; imports never install missing packages implicitly.
Non-npm `packageManager` declarations fail explicitly; pnpm/Yarn support is pending.

Inside model-authored eval:

```ts
await installPackages(['is-number@7.0.0']);
import isNumber from 'is-number';
return isNumber(value);
```

Static default, named and namespace imports and dynamic `await import(...)` are
supported for installed dependencies explicitly declared in the project's
`package.json`, including exported package subpaths. Undeclared/transitive
packages, local relative/absolute files, URL imports, and `node:` built-ins are
not importable from natlang eval or source functions. Imported package code may
itself use Node's normal module resolution internally.

Application subfunctions use the source tree, not import statements. `foo.nl` or
`foo.ts` automatically sees functions in `foo/`; `foo/bar.nl` sees functions in
`foo/bar/`. Runtime source imports between application files are rejected.
The native file format still requires a default function with explicit boundary
types. Recursive subfunction graphs are not admitted to unit-test training data.

Imported bindings are temporary within one eval and are not serialized into the
portable scope. Re-import in subsequent evals; the application installation persists.
The model's system prompt lists installed direct dependencies from `package.json`
(including scoped package names). The list refreshes between model turns after
`installPackages` succeeds, and is also supplied to child source-workspace calls.
Transitive or merely declared-but-not-installed packages are not advertised.
HTTP Response handles have the same lifetime. Consume them before returning:

```ts
const response = await fetch('https://example.com/data.json');
return await response.json();
```

Network access defaults on when a project manifest is found; `network: true` can also
enable fetch without one. `network: false` only removes
the provided fetch global—it cannot prevent installed Node packages from networking.

## Execution authority and reproducibility

This Node backend is a **trusted host backend, not a security sandbox**. Installed
packages, lifecycle scripts and imported Node builtins run with host process
permissions. Run untrusted applications in an appropriately isolated process or
container. A workspace directory does not itself provide that isolation.

Installation, filesystem and HTTP effects are not rolled back when eval fails.
Installation/import events and HTTP status events enter the host trace. HTTP logs
omit query strings, headers and bodies; they are observations, not replay fixtures.
Package-install records include manifest/lockfile hashes. npm has a 120-second
operation timeout; this is not a comprehensive descendant-process/resource limit.
Use AbortSignal timeouts for HTTP calls and outer job limits for execution.

Node caches imported modules. After upgrading an already-loaded package, restart
the execution process to guarantee the new module graph. Local `.ts` module loading
follows the host Node version's TypeScript support; published JS package entrypoints
work independently of that support. Direct HTTP module imports are not implemented;
HTTP requests through fetch and npm-supported dependency URLs are available.

See `examples/npm_app`: after `npm run build:node` in `ts-host`, run
`node examples/npm_app/run.mjs --install` from the repository root. It runs both the
handwritten and scripted natural-language versions against the same real npm package.
