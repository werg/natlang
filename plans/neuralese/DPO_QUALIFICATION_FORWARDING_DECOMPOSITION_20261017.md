# Forward explicit preference qualification and queue inactive stages

Owner policy: qualify the exact whole offline-preference objective on the restored optimizer before useful preference updates, and retain future stages until their admission boundary. Ordinary CE qualification behavior stays unchanged.

1. The recipe declares the preference source, objective and bound memory descriptor; this remains natural-language scheduling policy expressed in the existing recipe.
2. `continuation_config` is crisp command plumbing: retain an explicit preference memory descriptor, while ordinary CE continues to omit its one-time parent qualification.
3. `student_cycle` is crisp execution of declared admission: exclude stages explicitly marked `admitted: false`; the declared foundation stage must still exist.
4. Regressions exercise emitted native-preference CLI arguments and the actual mandatory-stage traversal. Existing prepared-objective qualification owns all numerical checks and checkpoint/RNG preservation.
