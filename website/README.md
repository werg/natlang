# natlang, explained

An interactive explainer of natlang for technically curious readers. It is a static site with no build step and no
dependencies beyond three Google Fonts. Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server -d website 8000   # then open http://localhost:8000
```

| File | Contents |
|---|---|
| `index.html` | The page: nine chapters, each opening with an informal inference rule |
| `style.css` | Colour tokens (amber is prose, cyan is crisp, violet is soft), light and dark themes, layout |
| `app.js` | The widgets. Each one is a small re-implementation of one rule from the specification |

The widgets model rules from [`spec/SPEC.md`](../spec/SPEC.md), its [extensions](../spec/ext/), the
[inline lambda design](../docs/inline-natlang-lambdas.md), [`iterateOn`](../docs/ITERATE_ON_PLAN.md) and
[trace specialization](../plans/TRACE_SPECIALIZATION.md). None of them calls a model. Where a widget shows a model's
answer, the page labels it illustrative. When the specification changes, update the chapter text and the widget that
models the rule; the specification wins any disagreement.
