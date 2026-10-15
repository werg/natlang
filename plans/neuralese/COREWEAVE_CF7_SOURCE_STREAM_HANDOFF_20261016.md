# Qualified arithmetic with corrected source readers

This operational runtime derives from qualified cf7c53d6019a. It imports the
shared stream_pending_batches implementation published in7422c09a7, changes only
native SFT cursor routing/cleanup, and mechanically applies source-safe stop
behavior from3bfd39626. All seven admission-cache policy files and every model,
attention, expert, precision and optimizer implementation remain exact cf7 bytes.
This avoids activating unqualified newer FP8/attention arithmetic during an
urgent IO/data continuation. It is the same two production lineages and shared
trainer mechanism; weights, moments, RNG, exposure and the fixed foundation
continue from exact current checkpoints. V6 uses an explicit append binding.

Thirty-five source-stream/native-SFT tests passed in141.38s, including actual
stop/update/resume and compressed source reader order/count behavior. Quick
check passed. This is not target qualification of newer FP8 or prefix sharing.
