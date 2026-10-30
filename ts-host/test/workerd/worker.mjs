// The worker the workerd conformance test runs: the vendored core bundle and the shared scenarios, with a realm over the
// UNSAFE_EVAL binding (workerd forbids code generation from strings everywhere else).
import * as core from '../../dist/core/natlang-core.js';
import { scenarios } from '../support/core-scenarios.mjs';

export default {
  async fetch(request, env) {
    const realm = core.constructorRealm((body, name, ...parameters) => env.UNSAFE_EVAL.newFunction(body, name, ...parameters));
    const results = {};
    for (const [name, run] of Object.entries(scenarios(core, realm))) {
      try { await run(); results[name] = 'ok'; }
      catch (error) { results[name] = String(error?.stack ?? error); }
    }
    return Response.json(results);
  },
};
