# Gemini CLINC static reference tranche (2026-10-15)

The free Gemini pool has 44 closed, source-disjoint CLINC plus/train batches of
256 assistant requests each. It produced 11,220 typed responses; 8,933 choices
exactly matched their pinned human source labels. The existing static decision
adapter initially rejected these cases because its provenance validator only
recognized PubMedQA source references and did not recognize the separate
license-provenance field. Commit `1465500d` added strict validation for the
pinned CLINC parquet/revision and recorded license facts. It did not change
the authored Natlang program or model-facing prompt.

The adapted 44 batches contain 1,137 multi-item programs with runtime-composed
inline `nl` decisions and 20,140 native decisions. All 1,137 scripted
references passed the runtime outcome and child capture-visibility checks.
They are authored static references: the provider supplies only typed child
choices, not scaffold code, reasoning or a trajectory. The source and label
checks reject 2,375 nonmatching/invalid rows before conversion. Independent
manual inspection of 190 unique source-matching request/label pairs found two
ambiguous or malformed requests; their entire eight-item programs are held.
The explicit semantic decision, held source IDs, complete adapter/source hashes,
and exact replay/turn checks are recorded under
`runs/free-provider-generation-20261013-v1/gemini-clinc-static-v2-{review,freeze}-20261015-v1`.

`freeze_gemini_clinc_static.py` is a crisp verifier and compactor. It checks
every selected source annotation, typed label, generated program receipt,
reference replay and native turn. The admission policy is the separate written
semantic review. The large native and replay JSONL streams are zstd-compressed
after their decoded hashes are pinned, recovering about 4 GiB on Pop. The
native rows keep `trace_admission=false`; the corpus registry admits only
authored static SFT. The renderer must use the recorded pending-review reason
override, exclude the two held programs, and apply the global protected,
held-source and quality filters.

The source corpus `gemini-clinc-static-reviewed-20261015-v1` and render
`gemini-clinc-static-reviewed-render-20261015-v1` are registered on main and
manifest-verified on DGX (`57e0104a`, `03c9b90b`). The render selected 20,104
native turns after the 36 turns from two ambiguous programs were held. Both
tokenizers have all 20,104 program-header documents and 10,052
whole-trajectory documents; the full view samples one per authored program
(1,018 documents). Its only
omissions are this declared full-view sampling. The next protected union must
still apply the global held-source and quality filters.

This tranche does not include batch45 or later. Batch46 closed with 195
source-matching labels and 25/25 exact scripted reference replays; it remains
held for a later semantic review tranche.
