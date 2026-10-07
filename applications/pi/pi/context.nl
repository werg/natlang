---
description: What a coding agent should know before it starts on task in the project at cwd. Returns the project's instructions (AGENTS.md or CLAUDE.md), the skills it can use, and the files to start from.
args:
  task: string
  cwd: string
  skillDirs: string[]
returns: ProjectContext
model: small
---
Gather what pi puts in its system prompt, and the scout's suggestions.

instructions: in cwd and every directory above it up to the root, the file AGENTS.md, or CLAUDE.md when there is
no AGENTS.md (files.read with the absolute path). List them outermost first, each with its path and its whole text.

skills: a skill is a directory holding a SKILL.md file. Look under ~/.pi/agent/skills, under .pi/skills in cwd, and
under each of skillDirs, up to four levels deep, and stop descending at a directory that holds a SKILL.md (a shell
find is fine). Read each SKILL.md's frontmatter, the block between the leading --- lines: name (else the directory's
name) and description. Skip a skill without a description, and one whose frontmatter says
disable-model-invocation: true. Keep each skill's location, the absolute path of its SKILL.md; list each once.

start: when the project has at least 12 files (files.list()), call scout(task, those files) and keep the paths it
gives that are among them, at most 8. With fewer files, start is empty.
