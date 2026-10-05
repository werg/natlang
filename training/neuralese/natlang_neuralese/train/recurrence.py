"""Per-reader sharing of deterministic writes in a verified producer DAG."""


def is_acyclic(dependencies):
    remaining = {name: set(children) & dependencies.keys() for name, children in dependencies.items()}
    ready = [name for name, children in remaining.items() if not children]
    parents = {name: set() for name in remaining}
    for name, children in remaining.items():
        for child in children:
            parents[child].add(name)
    seen = set()
    while ready:
        child = ready.pop()
        if child in seen:
            continue
        seen.add(child)
        for parent in parents[child]:
            remaining[parent].discard(child)
            if not remaining[parent]:
                ready.append(parent)
    return len(seen) == len(remaining)


class ProducerMemo:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.values = {}
        self.hits = 0

    def write(self, name, depth, compute):
        key = (name, depth)
        if self.enabled and key in self.values:
            self.hits += 1
            return self.values[key]
        value = compute()
        if self.enabled:
            self.values[key] = value
        return value


def curriculum_max_writes(curriculum, max_writes):
    """A sampled chain selects one edge at each nested level, without detaching it."""
    if curriculum == 'sampled-chain':
        if max_writes not in (0, 1):
            raise ValueError('sampled-chain requires max-writes 0 or 1')
        return 1
    return max_writes
