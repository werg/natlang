---
name: changelog-writer
description: Writes user-facing changelog entries from merged pull requests. Use when asked to draft release notes or update CHANGELOG.md.
license: MIT
---

# Changelog writer

1. Collect merged changes since the last tag (`scripts/collect.sh <tag>` lists them).
2. Group them under Added, Changed, Fixed and Removed.
3. Write each entry for users: what changed and why it matters, not how.
4. Follow the house style in `references/style.md`.
