"""Spike: the natlang module for Python. nl, iterate_on and reducers are the host's (the same objects eval uses); this
module gives them Python's shapes: nl[T] types a call, a Python type becomes the host's type text, and wait() runs one
in synchronous code (a pandas apply) through Pyodide's JSPI."""
import dataclasses, types, typing
from pyodide.ffi import create_proxy, run_sync
import _natlang_host as host

def type_text(t) -> str:
    """A Python annotation as natlang type text."""
    origin, args = typing.get_origin(t), typing.get_args(t)
    if t is bool: return 'boolean'
    if t in (int, float): return 'number'
    if t is str: return 'string'
    if t is type(None): return 'null'
    if t is typing.Any or t is None: return 'unknown'
    if origin is typing.Literal: return ' | '.join(repr(a).replace("'", '"') for a in args)
    if origin in (typing.Union, types.UnionType): return ' | '.join(type_text(a) for a in args)
    if origin in (list, tuple): return f'{type_text(args[0]) if args else "unknown"}[]'
    if origin is dict: return f'Record<string, {type_text(args[1])}>'
    if dataclasses.is_dataclass(t) or typing.is_typeddict(t):
        hints = typing.get_type_hints(t)
        return '{ ' + ', '.join(f'{k}: {type_text(v)}' for k, v in hints.items()) + ' }'
    raise TypeError(f'{t!r} has no natlang type')

class _Nl:
    """nl("instructions") is an open-result function; nl[T]("instructions") returns T."""
    def __init__(self, returns='unknown'): self.returns = returns
    def __getitem__(self, t): return _Nl(type_text(t))
    def __call__(self, instructions):
        return host.nl(instructions, self.returns)

nl = _Nl()

def wait(awaitable):
    """Run an nl call (or any awaitable) from synchronous code, such as a function pandas applies."""
    return run_sync(awaitable)

class _Iteration:
    """The host's iterateOn with a Python step: its progress reviews, limits and trace are the same as in eval. The
    step and the stopping check are lent to the host for the whole run (a plain Python callable passed to JavaScript
    lives only for the duration of the call), and released when it ends."""
    def __init__(self, step, initial, fixed): self.step, self.initial, self.fixed = step, initial, fixed
    async def until(self, done):
        step, check = create_proxy(self.step), create_proxy(done)
        try: return await host.iterate_on(step, self.initial, *self.fixed).until(check)
        finally: step.destroy(); check.destroy()

def iterate_on(step, initial, *fixed): return _Iteration(step, initial, fixed)
