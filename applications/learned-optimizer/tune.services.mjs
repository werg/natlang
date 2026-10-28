// The learned optimizer's tune services (ts-host/src/tune/services.ts): `natlang tune` runs whole `adapt` episodes as
// roots, so teacher collection, evaluation and RL rollouts reach the exact adaptation services. A root example names
// its prepared program context in `reference` (`python -m natlang_neuralese.tune.learned meta-examples` writes them):
// `{ plan, out }`. One persistent Python service serves every root; it keeps each context loaded across calls.
// The Python interpreter is `NATLANG_LOPT_PYTHON` (default python3).
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AdaptationServices, PythonService, optimizerDeclaration } from './python.mjs';
import { nativeBridge } from './native-bridge.mjs';

const repository = process.env.NATLANG_LOPT_REPOSITORY ?? resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
let python = null;

export const declarations = { optimizer: optimizerDeclaration };

export default function services(example) {
  const reference = example.reference ?? {};
  // Counterfactual decision roots are pure functions; only the whole episode needs a live adaptation context.
  if (example.function !== 'adapt.nl') return {};
  if (typeof reference.plan !== 'string' || typeof reference.out !== 'string')
    throw new Error(`root ${example.id} names no adaptation context; an adapt root carries reference { plan, out }`);
  python ??= new PythonService({ argv: [process.env.NATLANG_LOPT_PYTHON ?? 'python3', '-m'], cwd: repository,
    env: { PYTHONPATH: [join(repository, 'training/neuralese'), repository, process.env.PYTHONPATH].filter(Boolean).join(':') } });
  return { optimizer: new AdaptationServices(python, reference.plan, reference.out,
    nativeBridge(repository, python, reference.plan, reference.out)).optimizer };
}
