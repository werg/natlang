// Providers on the Models page: Pi's provider catalog (src/model/pi-provider.ts) with each provider's connection, its
// sign-in flows run from the page (subscription OAuth with its URL or device code, or the provider's API-key prompts;
// POST /api/providers/:id/login) and one-click profiles on its recommended model, the newest Luna or Haiku it serves
// (src/model/provider-recommendations.ts).

/** Providers shown first: the subscription sign-ins and the common API services. */
const FEATURED = ['anthropic', 'openai', 'github-copilot', 'openrouter', 'google', 'xai', 'mistral', 'groq', 'together', 'amazon-bedrock'];

export const connection = provider => provider.connected ?
  `${provider.connected.type === 'oauth' ? 'subscription' : 'API key'}${provider.connected.source ? ` · ${provider.connected.source}` : ''}` : null;

/** Save (or update) the profile named after the provider on `model`, optionally as the default. */
export async function useProvider(ui, provider, model, makeDefault) {
  await ui.api('PUT', `/api/profiles/${encodeURIComponent(provider.id)}`, { profile: { provider: provider.id, model }, makeDefault });
  ui.toast(`Profile ${provider.id} runs ${model}${makeDefault ? ' and is the default' : ''}`);
  ui.refreshRail();
}

/**
 * Run a provider's sign-in in a dialog. Resolves when the dialog closes: 'connected' when it stored a credential,
 * 'choose' when the user went on to pick a model (the caller's profile editor opens), else null.
 */
export function signIn(ui, provider, type) {
  const { h, fill } = ui;
  const body = h('div', { class: 'stack' }, h('div', { class: 'muted' }, 'Starting…'));
  const dialog = h('dialog', { class: 'login' }, h('h2', {}, type === 'oauth' ? provider.oauth.label ?? `Sign in: ${provider.oauth.name}` : `${provider.apiKey.name} for ${provider.name}`), body,
    h('div', { class: 'row' }, h('button', { type: 'button', onclick: () => dialog.close() }, 'Close')));
  document.body.append(dialog); dialog.showModal();
  let login, timer, shownPrompt = null, next = null;
  return new Promise(resolveSignIn => {
    dialog.addEventListener('close', () => {
      clearInterval(timer);
      if (login?.state === 'running') ui.api('DELETE', `/api/logins/${login.id}`).catch(() => {});
      dialog.remove(); resolveSignIn(login?.state === 'done' ? next ?? 'connected' : null);
    });
    const then = (action, value) => ui.attempt(async () => { await action?.(); next = value; dialog.close(); });
    const events = h('div', { class: 'stack' }), question = h('div', {}), state = h('div', {});
    fill(body, events, question, state);
    const draw = () => {
      fill(events, login.events.map(event => {
        if (event.type === 'auth_url') return h('div', { class: 'callout' }, h('div', {}, event.instructions ?? 'Open this page to sign in, then return here.'),
          h('div', { class: 'row' }, h('a', { class: 'button primary', href: event.url, target: '_blank', rel: 'noopener' }, 'Open sign-in page'),
            h('button', { type: 'button', onclick: () => navigator.clipboard?.writeText(event.url).then(() => ui.toast('Link copied')) }, 'Copy link')));
        if (event.type === 'device_code') return h('div', { class: 'callout' }, h('div', {}, 'Open ', h('a', { href: event.verificationUri, target: '_blank', rel: 'noopener' }, event.verificationUri), ' and enter'),
          h('div', { class: 'device-code mono' }, event.userCode));
        return h('div', { class: 'small muted' }, event.message, ...(event.links ?? []).map(link => [' ', h('a', { href: link.url, target: '_blank', rel: 'noopener' }, link.label ?? link.url)]));
      }));
      if (login.prompt?.id !== shownPrompt) {
        shownPrompt = login.prompt?.id ?? null;
        const prompt = login.prompt;
        if (!prompt) fill(question);
        else {
          const field = prompt.type === 'select' ? h('select', {}, prompt.options.map(option => h('option', { value: option.id }, `${option.label}${option.description ? ` — ${option.description}` : ''}`))) :
            h('input', { type: prompt.type === 'secret' ? 'password' : 'text', placeholder: prompt.placeholder ?? '', autocomplete: 'off' });
          fill(question, h('form', { class: 'stack', onsubmit: ui.attempt(async event => {
            event.preventDefault();
            login = await ui.api('POST', `/api/logins/${login.id}/answer`, { prompt: prompt.id, value: field.value }); draw();
          }) }, h('label', { class: 'field' }, prompt.message, field), h('div', { class: 'row' }, h('button', { type: 'submit', class: 'primary' }, 'Continue'))));
          field.focus();
        }
      }
      if (login.state === 'done') {
        // The page's provider list shares this object: its profile editor sees the new connection.
        provider.connected = { type, source: 'stored credential' }; provider.stored = type; provider.problem = null;
        const model = provider.recommended.model;
        fill(state, h('div', { class: 'callout ok' }, `Connected to ${provider.name}.`),
          h('div', { class: 'row' }, model ? h('button', { class: 'primary', onclick: then(() => useProvider(ui, provider, model, true)) }, `Use ${model} as the default`) : null,
            model ? h('button', { onclick: then(() => useProvider(ui, provider, model, false)) }, `Add a ${provider.id} profile`) : null,
            h('button', { class: model ? '' : 'primary', onclick: then(null, 'choose') }, 'Choose model…')));
      } else if (login.state === 'failed' || login.state === 'cancelled') fill(state, h('div', { class: 'callout warn' }, login.error ?? `Sign-in ${login.state}.`));
      else fill(state, login.prompt ? null : h('div', { class: 'small muted' }, 'Waiting for the provider…'));
      if (login.state !== 'running') clearInterval(timer);
    };
    ui.api('POST', `/api/providers/${encodeURIComponent(provider.id)}/login`, { type }).then(started => {
      login = started; draw();
      timer = setInterval(async () => { try { login = await ui.get(`/api/logins/${login.id}`); draw(); } catch (error) { fill(state, h('div', { class: 'callout warn' }, error.message)); clearInterval(timer); } }, 1000);
    }, error => fill(body, h('div', { class: 'callout warn' }, error.message)));
  });
}

/** The Providers panel. `ui` carries the page helpers; `onChange` redraws the page after a connection changes. */
export function providersPanel(ui, providers, onChange) {
  const { h, fill } = ui;
  const byId = new Map(providers.map(provider => [provider.id, provider]));
  const featured = FEATURED.map(id => byId.get(id)).filter(Boolean);
  const others = providers.filter(provider => !FEATURED.includes(provider.id)).sort((a, b) => a.name.localeCompare(b.name));
  const connectedOthers = others.filter(provider => provider.connected);
  const changed = () => onChange();
  const connect = async (provider, type) => {
    const outcome = await signIn(ui, provider, type);
    if (outcome === 'choose') ui.editProfile('', { provider: provider.id, model: provider.recommended.model });
    else if (outcome) changed();
  };
  const row = provider => {
    const model = provider.recommended.model, status = connection(provider);
    return h('tr', {},
      h('td', {}, h('strong', {}, provider.name), h('div', { class: 'small muted mono' }, provider.id)),
      h('td', {}, status ? h('span', { class: 'pill ok' }, status) : provider.problem ? h('span', { class: 'pill bad', title: provider.problem }, 'sign-in expired') : h('span', { class: 'pill idle' }, 'not connected')),
      h('td', { class: 'mono small' }, model ?? h('span', { class: 'muted' }, `${provider.models} models`)),
      h('td', {}, h('div', { class: 'row' },
        provider.oauth && (!provider.connected || provider.problem) ? h('button', { class: 'primary', onclick: () => connect(provider, 'oauth') }, provider.oauth.subscription ? 'Sign in with subscription' : 'Sign in') : null,
        provider.apiKey?.prompts && !provider.connected ? h('button', { class: provider.oauth ? '' : 'primary', onclick: () => connect(provider, 'api_key') }, 'Add API key') : null,
        provider.connected && model ? h('button', { class: 'primary', onclick: ui.attempt(async () => { await useProvider(ui, provider, model, true); changed(); }) }, `Use ${provider.recommended.model === provider.recommended.haiku ? 'Haiku' : provider.recommended.model === provider.recommended.luna ? 'Luna' : model} as default`) : null,
        provider.connected ? h('button', { onclick: () => ui.editProfile('', { provider: provider.id, model }) }, 'Choose model…') : null,
        provider.connected ? h('button', { onclick: () => ui.startJob({ kind: 'ask', instruction: 'Reply with the single word: ready', input: '', provider: provider.id, model }) }, 'Test') : null,
        provider.stored ? h('button', { class: 'danger', onclick: ui.attempt(async () => { await ui.api('DELETE', `/api/providers/${encodeURIComponent(provider.id)}/credential`); ui.toast(`Disconnected ${provider.name}`); changed(); }) }, 'Disconnect') : null)));
  };
  const pickedRow = h('tbody', {});
  const picker = h('select', { onchange: () => { const provider = byId.get(picker.value); fill(pickedRow, provider ? row(provider) : null); } },
    h('option', { value: '' }, `Another provider (${others.length})…`), others.map(provider => h('option', { value: provider.id }, `${provider.name}${provider.connected ? ' · connected' : ''}`)));
  const table = rows => h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Provider'), h('th', {}, 'Connection'), h('th', {}, 'Recommended model'), h('th', {}))),
    h('tbody', {}, rows.map(row)), pickedRow));
  return h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Providers'), picker),
    h('div', { class: 'small muted' }, 'Hosted models through Pi: sign in with a subscription (Claude Pro/Max, ChatGPT Plus/Pro, GitHub Copilot, …) or add an API key; keys from the providers\' environment variables are picked up too. "Use as default" saves a profile named after the provider on its newest Luna or Haiku model.'),
    table([...featured, ...connectedOthers]));
}
