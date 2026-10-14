# Backlog v5 teacher action-shape review (2026-10-14)

`luna-held-shapes-closed-backlog-v5-native-turns-20261014-v1` had already
admitted exact-oracle successes. Exact final answers do not certify that the
teacher followed the natlang process requested by the source program. Pop
scanned every approved train decision with
`scripts/neuralese_data/audit_teacher_action_shape.py` and pinned the result
in `BACKLOG_V5_SHAPE_REVIEW_20261014.json`. The source file SHA-256 is
`6ab5c37b559aa00c461ba253d2c241810950fa1e93dfb4fd0db27aaa94fa3cd2`.

Of 3,029 distinct parents, **614 are held** by exact action-shape checks:

| Source family | Held parents | Observed structural defect |
| --- | ---: | --- |
| composed process | 129 | Required `iterateOn` absent; many exact answers came from arbitrary capped loops. |
| edit stream | 101 | Required stepwise `iterateOn` and NL edit absent. |
| folder criteria reducer | 372 | Literal `.md` ticket selection with no NL criterion call. |
| workbench tickets | 12 | Required inline NL decision absent. |

The witness
`training/source-quarantines/backlog-v5-teacher-shape-20261014.jsonl`
(SHA-256 `5d87e164347a44c71fb7b56ab2b2cd289fcc012b7c3bad91c0c6bfa0606602ca`)
contains 1,136 source groups. Use it as a `--quarantine-reference` in every
target union that includes backlog-v5 source renders. The group-level exclusion
also removes 25 otherwise unflagged folder parents that share one of those
groups. A later source-level recuration and rerender can recover such siblings
after semantic review. Other backlog-v5 families remain candidates; this
structural scan is not their full quality certification.

The LFM and Mellum R8 teacher delta must include this witness in addition to
its existing protected held and source-quality references. Do not train from
the unfiltered backlog-v5 render. The currently running CoreWeave PH snapshot
does not include this proposed delta and needs no mutation.
