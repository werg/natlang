/**
 * The model a Pi provider profile defaults to: the newest Luna (OpenAI `gpt-N[.M]-luna`) or Haiku (Anthropic
 * `claude-haiku-N-M`) model in the provider's catalog. Versions are read from the catalog's own IDs, so a newer Pi
 * catalog moves the default without a code change; dated snapshots, `-pro`, `:batch` and `~…-latest` aliases are not
 * defaults because a profile should name a stable model. Anthropic's own providers default to Haiku, every other
 * provider to Luna when it serves one.
 */
export type ProviderRecommendation = { luna?: string; haiku?: string; model?: string };

const LUNA = /(?:^|\/)gpt-(\d+)(?:\.(\d+))?-luna$/;
const HAIKU = /(?:^|\/)claude-haiku-(\d+)[-.](\d+)$/;

function newest(ids: readonly string[], pattern: RegExp): string | undefined {
  let best: { id: string; major: number; minor: number } | undefined;
  for (const id of ids) {
    const match = pattern.exec(id);
    if (!match) continue;
    const major = Number(match[1]), minor = Number(match[2] ?? 0);
    if (!best || major > best.major || major === best.major && (minor > best.minor || minor === best.minor && id.length < best.id.length))
      best = { id, major, minor };
  }
  return best?.id;
}

export function recommendedProviderModel(provider: string, ids: readonly string[]): ProviderRecommendation {
  const luna = newest(ids, LUNA), haiku = newest(ids, HAIKU);
  const anthropicFirst = provider === 'anthropic' || provider.startsWith('anthropic-');
  const model = anthropicFirst ? haiku ?? luna : luna ?? haiku;
  return { ...(luna ? { luna } : {}), ...(haiku ? { haiku } : {}), ...(model ? { model } : {}) };
}
