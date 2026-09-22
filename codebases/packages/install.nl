import { publish } from "./install/publish";
import { rejected } from "./install/rejected";
import { validate } from "./install/validate";
---
description: Check a locked natlang bundle and publish it to an offline installation target.
args:
  lock: Lock
  target: Text
returns: InstallReport
---
function install(lock, target) -> InstallReport
  checked = validate(lock)
  if checked.ok:
    return publish(lock, target)
  return rejected(target, checked)
