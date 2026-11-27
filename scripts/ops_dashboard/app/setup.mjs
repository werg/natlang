// The workspace lifecycle for the natlang ops page (the starter's setup.mjs without its page templates: the page source
// is pages/ops/). One agent, one shared signer, one page:
//   npm run init -- ops      prepare the host's publisher policy and create the page's genesis and first epoch key (this agent is the creator)
//   npm run build -- ops     build pages/ops/dist/
//   npm run deploy -- ops    build and sign/publish the complete UI (an immutable release)
// The page itself is created once with `tendril page new ops --name "natlang ops"`; see ../README.md "Setting up".
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { requireSlug, workspace, workspaceRoot } from './shared/local.mjs';

const pageModule = (slug, name) => import(pathToFileURL(join(workspaceRoot, 'pages', slug, name)).href);

export async function run(action, args) {
  const { values, positionals } = parseArgs({ args, allowPositionals: true, strict: true, options: { 'owner-grant': { type: 'string' }, 'browser-first': { type: 'boolean' } } });
  const [slug] = positionals;
  const selected = await workspace(), context = await selected.pageContext(requireSlug(slug));
  if (action === 'init') {
    const result = await (await pageModule(slug, 'setup.mjs')).initializePage(context, { browserFirst: values['browser-first'] === true,
      ...(values['owner-grant'] === undefined ? {} : { ownerGrant: values['owner-grant'] }) });
    return { ...result, next: `npm run deploy -- ${slug}, then give the owner the page URL and keep the agent program (tendril agent run) running: it admits the owner's browsers.` };
  }
  if (action === 'build' || action === 'deploy') {
    const { build } = await import('esbuild');
    const output = await (await pageModule(slug, 'build.mjs')).buildPage(context, { bundle: build });
    if (action === 'build') return { built: output, next: `npm run deploy -- ${slug}` };
    return (await pageModule(slug, 'publish.mjs')).publishPage(context, { root: pathToFileURL(output.endsWith('/') ? output : output + '/') });
  }
  throw Error('Use init, build or deploy with the page slug (ops)');
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { const result = await run(process.argv[2], process.argv.slice(3)); if (result !== undefined) console.log(JSON.stringify(result, null, 2)); }
  catch (error) { console.error(`${error.code ?? error.name}: ${error.message}`); process.exitCode = 1; }
}
