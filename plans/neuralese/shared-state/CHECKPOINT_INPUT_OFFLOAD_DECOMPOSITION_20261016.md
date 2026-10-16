# Complete native graphs with checkpoint inputs in host memory

The admitted Search/OWL source contains intact graphs up to 87,996 LFM and
96,447 Mellum physical tokens, with up to 70,667 and 78,534 imported KV slots.
The batch token setting packs whole graphs; it is not a truncation boundary.

The supervisor's natural-language policy is: keep each complete graph, choose
the fastest memory mode that fits the actual device, and use host memory for
checkpoint inputs when device storage prevents useful training. A larger new
source must lead to memory support, rather than permanent exclusion at the
largest previously tested sequence length. Configuration declares this choice;
the transfer helper makes no source-admission or training-policy decisions.

1. The existing layer-major graph executor remains the one forward mechanism.
2. The existing port `activation_checkpoint` callable remains the integration
   interface. Wrap that callable, including its FP8 precision/RNG contexts.
3. Use PyTorch's `save_on_cpu` around checkpoint invocation. Its outer saved
   tensor hooks relocate checkpoint inputs; the nonreentrant checkpoint's inner
   hooks still replace internal activations with recomputation. Do not wrap the
   whole model in offload hooks or copy frozen weights for each projection.
4. Restore those exact inputs for backward. Preserve outputs, gradients, actor
   lifetime, optimizer/RNG, graph boundaries and hybrid-state imports. No token
   shortening, graph replay, weight/gradient approximation or model fork.
5. The shared trainer/recipe declares this optional mode and records it in the
   executable runtime capacity policy. CPU/offload capacity must be measured on
   the actual target before admitting the longest graphs to active sampling.

The crisp code is exact storage/transfer plumbing. A separate test compares
complete native outputs/gradients, checkpoint delegation, input storage and
precision contexts. Full target-model memory qualification belongs in the next
useful cloud runtime boundary; small local models do not establish that limit.

Primary API: [PyTorch saved tensor offload](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu).
