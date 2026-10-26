# natlang, explained

An interactive explainer of natlang for technically curious readers. It is a static site with no build step and no
dependencies beyond three Google Fonts. Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server -d website 8000   # then open http://localhost:8000
```

| File | Contents |
|---|---|
| `index.html` | The page: a home section and nine chapters. Each chapter opens with a poster plate, explains one part of the language and ends in a working exhibit |
| `style.css` | Flat poster palette with fixed meanings (mustard is prose, sky is types, plum is soft values, vermilion is the λ, actions and errors), light and dark themes, layout |
| `app.js` | The exhibits. Each one is a small re-implementation of one rule from the specification |

The widgets model rules from [`spec/SPEC.md`](../spec/SPEC.md), its [extensions](../spec/ext/), the
[inline lambda design](../docs/inline-natlang-lambdas.md), [`iterateOn`](../docs/ITERATE_ON_PLAN.md) and
[trace specialization](../plans/TRACE_SPECIALIZATION.md). None of them calls a model. Where a widget shows a model's
answer, the page labels it illustrative. When the specification changes, update the chapter text and the widget that
models the rule; the specification wins any disagreement.
