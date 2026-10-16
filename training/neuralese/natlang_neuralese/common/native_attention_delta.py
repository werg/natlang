"""Fixed per-row Flash-attention delta, independent of graph packing.

This corrects the compiler-generated reduction only. Attention forward and its
sparse tensor-core backward remain PyTorch's native Flex implementation.
"""
import functools
import types

import torch
import triton
import triton.language as tl


@triton.jit
def _delta_kernel(O, G, D, steps, heads, so0, so1, so2, so3, sg0, sg1, sg2, sg3,
                  WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = row % steps
    head = (row // steps) % heads
    batch = row // (steps * heads)
    d = tl.arange(0, BLOCK)
    o = tl.load(O + batch*so0 + head*so1 + token*so2 + d*so3, d < WIDTH, 0).to(tl.float32)
    g = tl.load(G + batch*sg0 + head*sg1 + token*sg2 + d*sg3, d < WIDTH, 0).to(tl.float32)
    tl.store(D + row, tl.sum(o*g, 0))


@torch.library.custom_op('natlang_native::attention_delta', mutates_args=(), device_types='cuda')
def attention_delta(out: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    result = torch.empty(out.shape[:-1], device=out.device, dtype=torch.float32)
    _delta_kernel[(result.numel(),)](out, grad, result, out.shape[2], out.shape[1],
        *out.stride(), *grad.stride(), WIDTH=out.shape[-1],
        BLOCK=triton.next_power_of_2(out.shape[-1]), num_warps=4, enable_fp_fusion=False)
    return result


@attention_delta.register_fake
def _fake_delta(out, grad):
    return out.new_empty(out.shape[:-1], dtype=torch.float32)


def verify_native_kv_gradient_allocations(source):
    """Bind names, order, shapes, strides and operand dtypes before interception."""
    import ast
    import textwrap
    expected = [
        ('layout_broadcasted_k', "FixedLayout(key.get_device(), key.get_dtype(), key_size, stride=[sympy.sympify(s) for s in key_strides])"),
        ('grad_query', "empty_strided(query_size, stride=[sympy.sympify(s) for s in grad_query_strides], dtype=query.get_dtype(), device=query.get_device())"),
        ('broadcasted_grad_value', "empty_strided(value_size, stride=[sympy.sympify(s) for s in value_strides], dtype=value.get_dtype(), device=value.get_device())"),
    ]
    tree = ast.parse(textwrap.dedent(source))
    top_level = tree.body[0].body
    actual = []
    signed_calls = []
    class Allocations(ast.NodeVisitor):
        def visit_Assign(self, node):
            value = node.value
            function = value.func if isinstance(value, ast.Call) else None
            name = function.id if isinstance(function, ast.Name) else getattr(function, 'attr', None)
            if name in {'FixedLayout', 'empty_strided'}:
                target = node.targets[0].id if node in top_level and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
                actual.append((target, ast.dump(value, include_attributes=False)))
                signed_calls.append(value)
            self.generic_visit(node)
    Allocations().visit(tree)
    signature = [(name, ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False))
                 for name, expression in expected]
    allocation_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and (node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None))
                        in {'FixedLayout', 'empty_strided'}]
    if actual != signature or {id(node) for node in allocation_calls} != {id(node) for node in signed_calls}:
        raise RuntimeError('native FP32 KV cotangents require review of exact Flex dK/dQ/dV allocation signatures')


_installed = False


def install_native_delta_lowering():
    """Select the fixed delta only for explicitly marked native Flex calls.

    A cloned lowering has a private lookup table: no temporary mutation of the
    global compiler registry while other models compile. Fail closed if PyTorch
    changes the exact delta expression we replace.
    """
    global _installed
    if _installed:
        return
    import inspect
    from torch._inductor.lowering import lowerings, make_fallback, register_lowering
    from torch._inductor.kernel.flex.flex_attention import flex_attention_backward
    original = inspect.unwrap(flex_attention_backward)
    source = inspect.getsource(original)
    expected = ('mul_delta = lowerings[aten.mul](out, grad_out)',
                'delta = lowerings[aten.sum](mul_delta, axis=-1)')
    if (any(source.count(line) != 1 for line in expected)
            or source.find('lowerings[aten.mul]') != source.find(expected[0]) + len('mul_delta = ')
            or source.find('lowerings[aten.sum]') != source.find(expected[1]) + len('delta = ')):
        raise RuntimeError('native attention delta requires review for this PyTorch lowering')
    op = torch.ops.natlang_native.attention_delta.default
    make_fallback(op)
    hop = torch.ops.higher_order.flex_attention_backward
    previous = lowerings[hop]

    @functools.wraps(original)
    def native_backward(*args, **kwargs):
        if not args[11].get('NATLANG_FIXED_DELTA', False):
            return previous(*args, **kwargs)
        args = list(args)
        args[11] = dict(args[11])
        args[11].pop('NATLANG_FIXED_DELTA')
        fp32_kv = args[11].pop('NATLANG_FP32_KV_GRADS', False)

        class DeltaLookup:
            pending = None

            def __getitem__(self, key):
                if key == torch.ops.aten.mul and self.pending is None:
                    def capture(out, grad):
                        self.pending = (out, grad)
                        return self.pending
                    return capture
                if key == torch.ops.aten.sum and isinstance(self.pending, tuple):
                    def reduce(pair, *, axis):
                        if pair is not self.pending or axis != -1:
                            raise RuntimeError('unexpected native attention delta reduction')
                        result = lowerings[op](*pair)
                        self.pending = False
                        return result
                    return reduce
                return lowerings[key]

        namespace = dict(original.__globals__, lowerings=DeltaLookup())
        if fp32_kv:
            verify_native_kv_gradient_allocations(source)
            fixed_layout = namespace['FixedLayout']
            empty = namespace['empty_strided']
            allocations = []
            def key_layout(device, dtype, *values, **options):
                return fixed_layout(device, torch.float32, *values, **options)
            def gradient_empty(*values, **options):
                allocations.append(None)
                if len(allocations) == 2:
                    options['dtype'] = torch.float32
                return empty(*values, **options)
            namespace.update(FixedLayout=key_layout, empty_strided=gradient_empty)
        fixed = types.FunctionType(original.__code__, namespace, original.__name__,
                                   original.__defaults__, original.__closure__)
        return fixed(*args, **kwargs)

    register_lowering(hop, type_promotion_kind=None)(native_backward)
    _installed = True
