"""Decoupled DiLoCo for the shared native trainers (ported from megatoken).

``protocol`` holds fragment planning and the outer optimizer, ``coordinator``
the durable HTTP synchronizer, ``worker`` the nonblocking learner, and
``cli`` the init/serve/status/export commands. Any trainer joins by exposing
a ``ParameterSurface`` and calling ``DilocoLearner.after_step`` after each
complete optimizer update. See plans/neuralese/DILOCO_COREWEAVE_20261027.md.
"""
