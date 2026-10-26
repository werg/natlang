# natlang, explained

An illustrated, interactive explanation of natlang for technically curious readers. It is a static site with no
build step and no dependencies beyond three Google Fonts.

| File | Contents |
|---|---|
| `index.html` | The page: an introduction with a complete example, then nine chapters. Each chapter opens with a poster plate, explains one part of the language and ends in a working exhibit |
| `style.css` | Flat poster palette with fixed meanings (mustard is prose, sky is types, plum is soft values, vermilion is the λ, actions and errors), light and dark themes, layout |
| `app.js` | The exhibits. Each one is a small re-implementation of one rule from the specification |
| `og.png` | The preview image that links to the site show, rendered from the page by `scripts/website-dev/social-card.mjs` |

The exhibits model rules from [`spec/SPEC.md`](../spec/SPEC.md), its [extensions](../spec/ext/), the
[inline lambda design](../docs/inline-natlang-lambdas.md), [`iterateOn`](../docs/ITERATE_ON_PLAN.md) and
[trace specialization](../plans/TRACE_SPECIALIZATION.md). None of them calls a model; where an exhibit shows a
model's answer, the page marks it as an example. When the specification changes, update the chapter text and the
exhibit that models the rule. The specification wins any disagreement.

## Run it locally and edit the text in place

```sh
npm run website:dev                # or: node scripts/website-dev/server.mjs
npm run website:dev -- --open      # also open it in the browser
```

The page is served at <http://127.0.0.1:4180/>. It reloads when a file in `website/` changes; stylesheet changes
apply without a reload.

To edit the copy, press **Edit text** in the toolbar at the bottom of the page and click any heading, paragraph, list
item, caption or label. Bold, italic and `code` are on the toolbar (also Ctrl+B, Ctrl+I and Ctrl+\`). Changed text is
outlined in mustard. **Save** (Ctrl+S or ⌘S) writes the changes into `website/index.html` and reloads the page where
you were; **Discard** restores the text you started with.

- Only the inner text of existing elements is written back; the rest of the file is left byte for byte as it was.
  The changes show up as an ordinary diff (`git diff website/index.html`).
- New paragraphs, list items, links and exhibits are added in `index.html` itself. Enter does nothing while editing.
- Elements with an `id`, code blocks, buttons and everything inside an exhibit are not editable, because the page's
  script fills or rewrites them. Add `data-no-edit` to an element to exclude it, or `data-edit` to include one the
  rules miss (the rules are in `scripts/website-dev/html-targets.mjs`).
- If `index.html` changes on disk while you have unsaved edits, saving is refused and the page says so, so that an
  edit made in the browser never overwrites one made in an editor.

Options: `--port <n>`, `--host <addr>`, `--open`, `--read-only` (live reload without the editing toolbar),
`--root <dir>` and `--page <file>`. Run with `--help` for details. The server binds to 127.0.0.1 by default and
accepts saves only from the page it serves.

The dev server's tests run with `npm run test:website`.

## Publish

The site is published at <https://werg.github.io/natlang/>. The workflow in
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) runs the tests on pull requests that touch the site,
and publishes `website/` to GitHub Pages whenever it changes on `main`. It can also be started by hand from the Actions
tab. Before the first run, set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**.

A typical round: run the dev server, edit and save, review `git diff website/`, commit, and merge to `main`. After
changing the headline or the example at the top of the page, render the preview image again with
`node scripts/website-dev/social-card.mjs`.

The page names its address in `og:url`, `og:image` and `rel="canonical"`. If the site moves, for example to a custom
domain, change those three to the new address.
