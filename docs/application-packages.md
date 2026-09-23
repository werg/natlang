# Application packages and network access

A Node-backed natlang application can own a normal `package.json` and
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

Construction requires an existing package.json. It does not install anything.
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
supported with an application workspace. Native Node ESM resolution loads installed
packages (including CommonJS packages), Node builtins and local modules. Relative
imports in eval resolve from the application root; relative imports in loaded
handwritten function files resolve from the source file. Existing native `.nl`/`.ts`
default-import companion-function behavior is retained. The native file format still
requires a default function with explicit boundary types; arbitrary library files
are imported as modules, not misinterpreted as native function files.

Imported bindings are temporary within one eval and are not serialized into the
portable scope. Re-import in subsequent evals; the application installation persists.
HTTP Response handles have the same lifetime. Consume them before returning:

```ts
const response = await fetch('https://example.com/data.json');
return await response.json();
```

Network access defaults on for an application workspace; `network: true` can also
enable fetch without a package workspace. The default environment without either
option retains its previous restricted eval surface. `network: false` only removes
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
