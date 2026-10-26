---
args: { progress: RetrofitProgress[], units: FieldedConfiguration[], applicability: ApplicabilityPartition }
returns: RetrofitFence
---
Compare each fielded unit's from and target configuration with its completed and verified retrofit steps. A partially complete unit remains fenced under its actual mixed state; mark complete only when every required step is verified for the target configuration.
