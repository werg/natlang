/* In-place copy editor and live reload for the website dev server (scripts/website-dev/server.mjs).
   Served only by that server; the published site never loads it. */
(() => {
  'use strict';
  const cfg = window.__NL_DEV__ || {};
  let version = cfg.version;
  let saving = false;
  let leaving = false;
  const session = {
    get(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* private mode: state is not kept */ } },
    del(k) { try { sessionStorage.removeItem(k); } catch (e) { /* ignore */ } },
  };

  /* ---------------------------------------------------------------- reload, keeping the reader's place */
  function reloadKeepingPlace() {
    leaving = true;
    session.set('nl-dev-scroll', String(scrollY));
    location.reload();
  }
  const savedScroll = session.get('nl-dev-scroll');
  if (savedScroll !== null) {
    session.del('nl-dev-scroll');
    const restore = () => scrollTo(0, Number(savedScroll));
    restore();
    addEventListener('load', () => requestAnimationFrame(restore));
  }

  /* ---------------------------------------------------------------- toolbar */
  const bar = document.createElement('div');
  bar.id = 'nl-dev';
  bar.setAttribute('role', 'toolbar');
  bar.setAttribute('aria-label', 'Page editing');
  bar.innerHTML = cfg.editable
    ? `<button type="button" data-act="edit" aria-pressed="false">Edit text</button>
       <span class="nl-fmt" hidden>
         <button type="button" data-act="bold" title="Bold (Ctrl+B)"><b>B</b></button>
         <button type="button" data-act="italic" title="Italic (Ctrl+I)"><i>I</i></button>
         <button type="button" data-act="code" title="Code (Ctrl+\`)"><code>code</code></button>
       </span>
       <span class="nl-status" aria-live="polite">Live reload is on</span>
       <button type="button" data-act="discard" hidden>Discard</button>
       <button type="button" data-act="save" class="nl-primary" hidden>Save</button>`
    : '<span class="nl-status">Read-only · live reload is on</span>';
  document.body.appendChild(bar);
  const $ = sel => bar.querySelector(sel);
  const status = (text, kind = '') => { const s = $('.nl-status'); s.textContent = text; s.dataset.kind = kind; };

  const banner = document.createElement('div');
  banner.id = 'nl-dev-banner';
  banner.hidden = true;
  document.body.appendChild(banner);
  function showBanner(text, actions) {
    banner.innerHTML = '';
    const p = document.createElement('span');
    p.textContent = text;
    banner.appendChild(p);
    for (const [label, fn] of actions) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.addEventListener('click', fn);
      banner.appendChild(b);
    }
    banner.hidden = false;
  }

  /* ---------------------------------------------------------------- live reload */
  const events = new EventSource('/__dev/events');
  events.addEventListener('change', e => {
    const d = JSON.parse(e.data);
    if (d.kind === 'css') {
      for (const link of document.querySelectorAll('link[rel="stylesheet"]')) {
        const url = new URL(link.href, location.href);
        if (url.origin === location.origin && url.pathname.endsWith('/' + d.file)) {
          url.searchParams.set('v', Date.now());
          link.href = url.href;
        }
      }
      return;
    }
    if (saving || leaving) return;
    if (d.file === cfg.page && d.version === version) return;
    if (cfg.editable && dirty().length) {
      showBanner(`${d.file} changed on disk. Your unsaved edits cannot be saved over it.`, [
        ['Reload and lose my edits', () => { leaving = true; reloadKeepingPlace(); }],
        ['Keep the edits on screen', () => { banner.hidden = true; status('Copy your edits elsewhere, then reload', 'error'); }],
      ]);
      return;
    }
    reloadKeepingPlace();
  });

  if (!cfg.editable) return;

  /* ---------------------------------------------------------------- editing */
  const targets = [...document.querySelectorAll('[data-nl-edit]')];
  const original = new Map(targets.map(el => [el, el.innerHTML]));
  let editing = false;

  /** The edited inner HTML, without artefacts the browser adds while typing. */
  function clean(el) {
    const orig = original.get(el);
    let html = el.innerHTML;
    html = html.replace(/<span style="[^"]*">([\s\S]*?)<\/span>/g, '$1').replace(/<\/?font[^>]*>/g, '');
    html = html.replace(/(<br\s*\/?>\s*)+$/, '');
    if (!orig.includes('&nbsp;')) html = html.replace(/&nbsp;/g, ' ');
    return html;
  }
  const dirty = () => targets.filter(el => clean(el) !== original.get(el));

  function refresh() {
    const changed = new Set(dirty());
    for (const el of targets) el.classList.toggle('nl-dirty', changed.has(el));
    const n = changed.size;
    $('[data-act="save"]').hidden = !editing && !n;
    $('[data-act="discard"]').hidden = !n;
    $('[data-act="save"]').disabled = !n;
    status(n ? `${n} unsaved change${n === 1 ? '' : 's'}` : editing ? 'Click any text to edit it' : 'Live reload is on', n ? 'dirty' : '');
  }

  function setEditing(on) {
    editing = on;
    document.body.classList.toggle('nl-editing', on);
    for (const el of targets) {
      if (on) { el.contentEditable = 'true'; el.spellcheck = true; }
      else el.removeAttribute('contenteditable');
    }
    $('[data-act="edit"]').setAttribute('aria-pressed', String(on));
    $('[data-act="edit"]').textContent = on ? 'Stop editing' : 'Edit text';
    $('.nl-fmt').hidden = !on;
    session.set('nl-dev-editing', on ? '1' : '0');
    refresh();
  }

  async function save() {
    const changed = dirty();
    if (!changed.length || saving) return;
    saving = true;
    status('Saving…');
    try {
      const res = await fetch('/__dev/save', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-natlang-dev': '1' },
        body: JSON.stringify({ version, edits: changed.map(el => ({ id: Number(el.dataset.nlEdit), html: clean(el) })) }),
      });
      const body = await res.json().catch(() => ({}));
      if (res.ok) {
        version = body.version;
        status(`Saved ${body.saved} change${body.saved === 1 ? '' : 's'} to ${cfg.page}`, 'ok');
        setTimeout(reloadKeepingPlace, 350);
        return;
      }
      if (res.status === 409) {
        showBanner(body.error, [['Reload and lose my edits', () => { leaving = true; reloadKeepingPlace(); }], ['Close', () => { banner.hidden = true; }]]);
      }
      status(body.error || `Saving failed (${res.status})`, 'error');
    } catch (e) {
      status('The dev server is not responding. Is it still running?', 'error');
    }
    saving = false;
  }

  function discard() {
    for (const el of dirty()) el.innerHTML = original.get(el);
    refresh();
  }

  function toggleCode() {
    const sel = getSelection();
    if (!sel.rangeCount) return;
    const range = sel.getRangeAt(0);
    const host = range.commonAncestorContainer.nodeType === 1 ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
    if (!host.closest('[data-nl-edit][contenteditable]')) return;
    const inside = host.closest('code');
    if (inside && inside.closest('[data-nl-edit]') === host.closest('[data-nl-edit]')) {
      inside.replaceWith(...inside.childNodes);
    } else if (!range.collapsed) {
      const code = document.createElement('code');
      code.appendChild(range.extractContents());
      range.insertNode(code);
      sel.removeAllRanges();
      const after = document.createRange();
      after.selectNodeContents(code);
      sel.addRange(after);
    }
    refresh();
  }

  bar.addEventListener('mousedown', e => { if (e.target.closest('.nl-fmt')) e.preventDefault(); }); // keep the text selection
  bar.addEventListener('click', e => {
    const act = e.target.closest('button')?.dataset.act;
    if (act === 'edit') setEditing(!editing);
    else if (act === 'save') save();
    else if (act === 'discard') discard();
    else if (act === 'bold' || act === 'italic') { document.execCommand(act); refresh(); }
    else if (act === 'code') toggleCode();
  });

  document.addEventListener('input', e => { if (e.target.closest?.('[data-nl-edit]')) refresh(); });
  document.addEventListener('keydown', e => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); save(); return; }
    const inEditable = e.target.closest?.('[data-nl-edit][contenteditable]');
    if (!inEditable) return;
    if (mod && e.key === '`') { e.preventDefault(); toggleCode(); }
    else if (e.key === 'Enter') { e.preventDefault(); status('New paragraphs and list items are added in the HTML file', 'hint'); }
    else if (e.key === 'Escape') { inEditable.blur(); }
  });
  document.addEventListener('paste', e => {
    if (!e.target.closest?.('[data-nl-edit][contenteditable]')) return;
    e.preventDefault();
    const text = (e.clipboardData.getData('text/plain') || '').replace(/\s+/g, ' ');
    document.execCommand('insertText', false, text);
  });
  // While editing, a click inside editable text places the caret instead of following a link.
  document.addEventListener('click', e => {
    if (editing && e.target.closest('[data-nl-edit]') && e.target.closest('a')) e.preventDefault();
  }, true);
  addEventListener('beforeunload', e => {
    if (!leaving && dirty().length) { e.preventDefault(); e.returnValue = ''; }
  });

  if (session.get('nl-dev-editing') === '1') setEditing(true);
  else refresh();
})();
