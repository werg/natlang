ID C_TARGET. Controlling protocol TARGET for this fictional device trial.
[C_Q] Question: what percentage of attempted signal detections succeed in the defined launch cohort and commissioning-relative interval?
[C_POP] Include every unit whose first accepted commissioning event is at or before global time0. First telemetry availability is not commissioning eligibility. A device can belong even if it first reports later.
[C_ANCHOR] Anchor each included unit at its first accepted commissioning event. Rejected commissioning attempts and later calibration do not move the origin.
[C_WINDOW] Count attempts at offsets0 through2 days relative to that anchor, inclusive at both ends.
[C_METRIC] Pool all included attempts; success percentage is100 times the number of pass outcomes divided by all included attempts. A pass is a successful signal detection. A fail is an unsuccessful detection. An empty denominator yields null, never0.
Population, anchor, window and metric meanings are independent roles. Exact baseline percentage equality does not establish semantic equivalence.
