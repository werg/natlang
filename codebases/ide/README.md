# Natlang IDE workbench

`edit.nl` interprets a user edit into an exact revisioned text patch. Invalid
source remains editable. `run.nl` checks the current revision and starts a
child source graph, retaining its trace and result. `view.nl` asks natlang to
compose source, diagnostic and trace panels; a crisp renderer escapes them to
HTML. Moving through recorded trace events never reexecutes the programme.

The host can freeze scenarios against source revisions and evaluate them with
an exact expected value. This is an IDE backend and generated view, not a
browser editor or a training service. It deliberately distinguishes a source
revision from each run's source graph revision and stores full traces outside
the natlang value flow.
