# Evidence atlas

`answer.nl` asks natlang to plan searches, select relevant hits and compose a
claim-by-claim answer. Search and reads are exact host operations over a
versioned collection. Each claim carries a source span ID, source revision and
short literal quote. The final crisp check verifies that every cited passage
is still an authentic span from the pinned collection and that its quote
appears there. It labels the result `citation-checked`, not semantically
proven; a separate reviewer must judge whether each quote supports its claim.

`EvidenceCollection` indexes local documents by paragraph, records offsets
and content digests, offers token-overlap search with explicit hit counts and
truncation, and rejects reads or verification after a collection revision
changes. Native documents and indexes stay in the host. No vector database or
global natlang search tool is needed for this first corpus.

The integration test runs the complete natlang search/read/compose sequence.
Adversarial tests reject invented quotes, modified passages and stale reads.
The model callback is scripted; live-model retrieval recall and grounded
answer quality remain open. The next useful corpus includes natlang source and
trace records, with trace coverage limits carried into answer gaps.
