---
description: Summarize a coding agent's conversation into a checkpoint it can continue from.
args:
  previous: string
  conversation: string
returns: string
model: small
---
conversation is the part of a coding agent's session that will leave its context; previous is the checkpoint
summary of what came before it (empty at first). Write the new checkpoint, keeping everything previous still holds
and adding what conversation adds, in exactly these sections: ## Goal; ## Constraints & Preferences; ## Progress
with ### Done (- [x] items), ### In Progress (- [ ] items) and ### Blocked; ## Key Decisions (**decision**:
rationale); ## Next Steps (numbered); ## Critical Context. Keep each section short and preserve exact file paths,
function names, commands and error messages.
