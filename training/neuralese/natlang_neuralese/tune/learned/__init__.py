"""The joint learned optimizer for native soft prompts, soft skills and LoRA mixtures
(plans/neuralese/JOINT_LEARNED_OPTIMIZER.md). Crisp mechanism only; policy lives in the Natlang program
`ts-host/src/tune/optimizer/`. See DECOMPOSITION.md."""


class LearnedOptimizerError(ValueError):
    pass
