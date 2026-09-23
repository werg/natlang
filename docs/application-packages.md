# Packages and network access in natlang code

Natlang code imports packages the way any module in its workspace would. The
workspace is the application directory: `createNatlangRuntime({ workspace })`,
`natlang run --workspace DIR`, or by default the nearest `package.json` above
the working directory. Install dependencies there with your package manager as
usual; the runtime never installs anything.

- Callable-folder TypeScript (`main/is_numeric.ts`) imports packages with normal
  `import` statements.
- Model-written eval code can use `import x from "pkg"` or `await import("pkg")`.
  Imported bindings last for that eval; import again in a later eval.
- Resolution is Node's, from the workspace, so anything the workspace can
  resolve is importable, including subpath exports.
- Application functions come from callable folders, not imports: `foo.nl` sees
  the items in `foo/`. Relative and absolute file imports from callable-folder
  code are rejected.

`fetch` is available in eval, and `network: false` removes it. HTTP responses
have to be consumed before the eval returns:

```ts
const response = await fetch('https://example.com/data.json');
result = await response.json();
```

## Authority

The Node backend runs code with the host process's permissions; it is not a
sandbox. Package code, filesystem access and HTTP requests are not undone when
an eval fails. Package imports and HTTP requests (origin and path only) are
recorded in the host trace as observations, not replay fixtures. Node caches
loaded modules; restart the process after upgrading a package that is already
loaded.

See `examples/npm_app`: run `npm install` in that directory, build `ts-host`,
then `node examples/npm_app/run.mjs`. It uses one npm package from handwritten
callable-folder code and from a scripted natural-language eval.
