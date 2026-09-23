# Natlang task board

The board is a browser-local application written in TypeScript
(`program/board.ts`). Its `reduce` function interprets a UI event with an inline
`nl` call and applies the decision exactly. Its `view` function asks natlang to
group and label the tasks; the layout helper verifies that each task appears
exactly once and emits a safe DOM tree. The page compiles the module with the
natlang compiler, an `EventLoop` orders events, and `BrowserDomRenderer` turns
the tree into controls that emit the next events.

From `ts-host`, run `npm run build`, then
`node scripts/serve-browser-local.mjs --port=8765`. Open
`http://127.0.0.1:8765/ts-host/examples/browser-board/` and load one of the
local model checkpoints. The model file and template must be available under
`/models/`. A `?fixture` query parameter uses scripted decisions for wiring
tests; it does not assess the model's semantic ability.

Task state is currently in browser memory and resets on reload. A production
board should add persisted event/state snapshots, model-quality scenarios and
loading/error affordances. The static shell uses HTML/CSS/JavaScript for model
loading and presentation; natlang calls in the module do the event interpretation and view planning.
