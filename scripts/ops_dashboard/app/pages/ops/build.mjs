// Build the publishable page graph into pages/ops/dist/: the dashboard UI (ui/), the owning session Worker and public
// configuration, bundled with esbuild. `npm run build -- ops` builds it, `npm run deploy -- ops` builds and publishes it
// (setup.mjs). Only this dist/ is ever published, never the workspace or `.tendriller/`.
import { copyFile, mkdir, rm, writeFile } from 'node:fs/promises';
import { execSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

/** The repository commit the page was built from ('' outside a checkout, as in a copied test workspace). */
const commitOf = cwd => { try { return execSync('git rev-parse --short=10 HEAD', { cwd, stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { return ''; } };

export async function buildPage(context, { root = new URL('./', import.meta.url), bundle, out } = {}) {
  const output = out ?? new URL('./dist/', root);
  await rm(output, { recursive: true, force: true });
  try {
    const prepared = await context.readPrivate('prepared.json');
    if (prepared.applicationId !== context.page.applicationId || prepared.publisherKey !== context.signer.publicKey)
      throw Error('Resume preparation for this exact workspace page (npm run init -- ops)');
    const pageUrl = new URL(context.page.pageUrl);
    if (!['https:', 'http:'].includes(pageUrl.protocol) || pageUrl.username || pageUrl.password || pageUrl.search || pageUrl.hash) throw Error('Invalid public page URL');
    await writeBundle({ root, output, bundle, configuration: { applicationId: context.page.applicationId,
      page_url: context.page.pageUrl, serviceOrigin: context.service, pageId: context.page.id, agentKey: context.signer.publicKey } });
    return fileURLToPath(output);
  } catch (cause) {
    await rm(output, { recursive: true, force: true });
    throw new Error(`Page build failed; no publishable dist/ remains. ${cause.message}`, { cause });
  }
}

/** The static graph: configuration, the UI files, the build stamp and the bundled page and Worker. */
async function writeBundle({ root, output, bundle, configuration }) {
  await mkdir(output, { recursive: true });
  await writeFile(new URL('configuration.json', output), JSON.stringify(configuration, null, 2) + '\n');
  for (const f of ['index.html', 'style.css']) await copyFile(new URL(`./ui/${f}`, root), new URL(f, output));
  // Build stamp: shown in the page footer and announced by the agent program's ticks, so an open tab that is still on an
  // older publication (Tendriller pins a tab, reload included) can move itself to the selected release.
  const stamp = { build: new Date().toISOString().replace(/\.\d+Z$/, 'Z'), commit: commitOf(fileURLToPath(root)) };
  await writeFile(new URL('build.json', output), JSON.stringify(stamp) + '\n');
  await bundle({ define: { __OPS_BUILD__: JSON.stringify(stamp) }, absWorkingDir: fileURLToPath(root),
    entryPoints: { page: 'ui/page.mjs', worker: 'worker.mjs' }, outdir: fileURLToPath(output),
    bundle: true, splitting: true, format: 'esm', platform: 'browser', target: 'es2022', outExtension: { '.js': '.mjs' },
    chunkNames: 'chunks/[name]-[hash]', assetNames: 'assets/[name]-[hash]', loader: { '.wasm': 'file' }, logLevel: 'warning' });
}

/** A graph for the local layout preview (preview_server.py): no workspace, no identity, never published. */
export async function buildPreview(out, { bundle } = {}) {
  const output = pathToFileURL(out.endsWith('/') ? out : out + '/'), root = new URL('./', import.meta.url);
  await rm(output, { recursive: true, force: true });
  await writeBundle({ root, output, bundle, configuration: { preview: true } });
  return fileURLToPath(output);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href && process.argv[2] === '--preview') {
  const { build } = await import('esbuild');
  console.log(JSON.stringify({ built: await buildPreview(process.argv[3], { bundle: build }) }));
}
