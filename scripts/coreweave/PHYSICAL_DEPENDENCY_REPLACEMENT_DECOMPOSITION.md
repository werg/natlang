# Exact inherited dependency replacement

The deployment policy chooses an immutable donor and reviewed overlay. This
change adds no policy choice. The existing installer remains crisp durability
and source verification plumbing.

1. Verify every declared donor dependency at its actual original path before
   creating a destination, as before.
2. Clone the physical tree and apply the sealed overlay, as before.
3. For each dependency, compare its relative path with the overlay's exact file
   inventory. An explicitly overlaid file must match that sealed replacement;
   record both the original donor binding and replacement binding. Every other
   dependency must retain its original exact bytes.
4. Run the existing package, role authority and publication import checks.

The replacement is not an inherited-byte claim. Missing, changed or unsealed
replacement bytes fail closed. No model, checkpoint decode or owner is started.
