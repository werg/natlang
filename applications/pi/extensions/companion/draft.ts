/**
 * The companion beside the user's draft (plans/STREAMING.md §3): a draft helper (host/drafts.ts) that offers relevant
 * context, warnings and clarifying questions while the user types. Its policy is a pluggable hot path: `crisp` checks
 * the paths and quoted names the draft writes against the workspace (a named file that does not exist is a warning; a
 * file the companion knows, or a symbol's definition, is context); `nl` is `offers.nl`, which reads the draft and the
 * recent conversation and searches the workspace; `shadow` runs both and uses the natural-language offers. The helper
 * only reads: offers are documents the UI shows, and nothing reaches the agent unless the user takes it.
 */
import { pluggable, pluggableMode, type NatlangRuntime, type PluggableSetting } from 'natlang:runtime';
import type { DraftHelper, DraftInput } from '../../host/drafts.ts';
import type { DraftOffer } from '../../types.ts';
import { hintsOf } from './stream.ts';
import { COMPANION_READ_DECLARATION, companionService, definitionPattern, renderMessage, workspacePath } from './workspace.ts';
import offers from './offers.nl';

/** Offers one pass returns. */
const MAX_OFFERS = 3;

export type CompanionDraftOptions = {
  /** The offer policy: `crisp` (default), `nl` (`offers.nl`) or `shadow` (both, the natural-language offers used). */
  offers?: PluggableSetting;
};

/** The crisp offers for a draft: what the paths and names it writes are in the workspace. */
async function crispOffers(input: DraftInput, service: ReturnType<typeof companionService>): Promise<DraftOffer[]> {
  const found: DraftOffer[] = [];
  const named = hintsOf(input.text);
  for (const path of named.files) {
    const local = workspacePath(input.cwd, path);
    if (local === undefined) continue;
    const file = await service.file(local);
    if (!file) found.push({ kind: 'warning', text: `${local} does not exist in the workspace.` });
    else if (file.known) found.push({ kind: 'context', text: `${local}: ${file.known.purpose}`, insert: `Relevant: ${local} (${file.known.purpose})` });
  }
  for (const symbol of named.symbols) {
    if (!/^[A-Za-z_$][\w$]*$/.test(symbol)) continue;
    const [line] = await service.search(definitionPattern(symbol));
    if (line) found.push({ kind: 'context', text: `\`${symbol}\` is defined at ${line}`, insert: `Relevant: \`${symbol}\` is defined at ${line}` });
  }
  return found.slice(0, MAX_OFFERS);
}

/** The companion's draft helper; its natural-language policy runs on `natlang`. */
export function companionDraftHelper(natlang: NatlangRuntime, options: CompanionDraftOptions = {}): DraftHelper {
  const mode = pluggableMode(options.offers, 'crisp');
  return {
    name: 'companion',
    async offers(input) {
      const service = companionService(input.read, input.context, input.env, input.cwd);
      const crisp = () => crispOffers(input, service);
      if (mode === 'crisp') return crisp();
      const recent = input.recent.map(renderMessage).filter(Boolean).join('\n');
      const choose = pluggable({ crisp, nl: async () => (await offers(input.text, recent) as DraftOffer[]).slice(0, MAX_OFFERS) },
        mode, { name: 'companion.offers' });
      return await natlang.run(choose, { services: { companion: service }, serviceDeclarations: { companion: COMPANION_READ_DECLARATION },
        signal: input.signal, name: `companion.offers#${input.conversationId}.${input.version}` }) as DraftOffer[];
    },
  };
}
