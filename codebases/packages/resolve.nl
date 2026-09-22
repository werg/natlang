import { choose } from "./resolve/choose";
import { finalize } from "./resolve/finalize";
import { solutions } from "./resolve/solutions";
import { unresolved } from "./resolve/unresolved";
---
description: Choose a compatible content-pinned package lock from exact offline solver results.
args:
  request: PackageRequest
returns: Resolution
---
function resolve(request) -> Resolution
  locks = solutions(request)
  if locks is empty:
    return unresolved(request)
  selected = choose(request, locks)
  return finalize(request, locks, selected)
