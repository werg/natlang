# Natlang task board

The board is a browser-local application whose reducer and view planner are
natlang programs in `program/board/`. `reduce.nl` interprets a UI event and
calls a crisp state transition helper. `view.nl` asks natlang to group and
label the tasks; the layout helper verifies that each task appears exactly
once and emits a safe DOM tree. `BrowserNatlangApplication` orders events and
`BrowserDomRenderer` turns the tree into controls that emit the next events.

From `ts-host`, run `npm run build`, then
`node scripts/serve-browser-local.mjs --port=8765`. Open
`http://127.0.0.1:8765/ts-host/examples/browser-board/` and load one of the
local model checkpoints. The model file and template must be available under
`/models/`. A `?fixture` query parameter uses scripted decisions for wiring
tests; it does not assess the model's semantic ability.

Task state is currently in browser memory and resets on reload. A production
board should add persisted event/state snapshots, model-quality scenarios and
loading/error affordances. The static shell uses HTML/CSS/JavaScript for model
loading and presentation; natlang owns event interpretation and view planning.
