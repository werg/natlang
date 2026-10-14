# Inline labeled-source contradiction audit (Pop, 2026-10-14)

The s52052 Luna closeout contains 90 exact-oracle accepted parents and 11
rejections. Four rejections are dataset-workbench decisions. Inspection found
a source problem distinct from model error: the pinned SST-2 cache labels
`a damn fine and a truly distinctive and a deeply pertinent film` **negative**
at row `idx=41765`, while the same phrase with a final period is **positive**
at `idx=55633`. The task asks whether a visible comment speaks badly of the
film, so an exact label for the negative copy is indefensible from the visible
text. The cached converted JSONL is SHA-256
`9ee25cb712f8c50c34ea6465de28f0bface80f2aee6663e19dd53085de5d8e4f`.
The conversion follows the pinned Hugging Face class labels; the source
annotation, not its index-to-name mapping, conflicts.

Future inline labeled cases now filter every row whose visible text has
opposing source labels after case folding and punctuation/space normalization,
before the train/test split. Among rows passing the existing length filters,
the pinned caches contain 161 SST-2, 54 AG News, zero SMS Spam and two
Banking77 rows with this contradiction. This narrow filter leaves all
unambiguous rows. It does not change the exact oracle or relabel old rows.
The next source build uses `labeled.mjs` from current main; already sealed
campaigns retain their source snapshots.

An exact source-record-ID scan of the 90 accepted s52052 parents found no
affected program. Of the 3,029 accepted parents in the previously admitted
closed-backlog v5, exactly one contains a conflicting record:
`inline-curriculum:s52040:dataset_workbench:dataset7:a`. Hold that program
and its source group from new renders and unions. DGX was asked to inspect
whether its current immutable union already includes it and to omit it from
the corrected next union; published manifests are not silently rewritten.

Other s52052 rejections remain separate: three Musique answers contain the
correct short gold phrase inside an overlong sentence, and a ham-labeled
personal referral text can plausibly satisfy the task's promotion/scam
criterion. These are review candidates, not automatic negatives or exact
teacher successes. A teacher correction may shorten the Musique final answers
with exact source verification; the ambiguous routing row should be held
before using its strict label as a training target.
