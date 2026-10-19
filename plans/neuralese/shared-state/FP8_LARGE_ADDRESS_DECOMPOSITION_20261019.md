# Large routed-gradient addresses

The recipe continues whole graphs; source size is not a reason to discard a
pending trajectory. Address computation is exact crisp kernel mechanism.

1. Preserve the actual failed 161972-token graph and original failure reports.
2. Calculate tensor addresses in signed 64-bit integers before multiplying row
   indexes by strides. Routed rows times width2304 exceed signed32 range for
   this graph, although each separate index and stride fits signed32.
3. Keep the dot products, tile boundaries, masks, reduction order and output
   casts unchanged. Apply the same law to ordinary and chunk-backed expert dW.
4. Compare actual small-domain gradients bitwise with the historical kernel.
   Reproduce the old negative address safely without dereferencing it, and run
   the repaired real dW kernel at the actual large stride/row address with a
   bounded strided tensor. No illegal memory access is needed to test the fix.
5. Build the urgent cloud repair from the sealed R9 plus only this address hunk;
   preserve old evidence as old evidence, and qualify the actual repaired runtime
   and complete pending graph before production. Later optimizations remain
   separately identified rather than silently mixed into this repair.
