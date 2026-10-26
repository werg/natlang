# Release compatibility coordinator

Compatibility is evaluated per behavioral change and consumer. The review treats default, ordering, retry, and error changes as externally meaningful even when a schema stays fixed. Consumer dependencies are supported by behavior evidence; uncertain consumers remain in replay scope, and unowned integrations block risky migration.

Paired replays use identical inputs and separate acceptance criteria for baseline and candidate. Migration waves retain old-version support windows and require observed rollback rehearsals that account for candidate-era writes. Readiness also checks that documentation and notices name the exact candidate revision; early pages remain explicitly candidate-only. This is a release proposal, not a migration dispatch.
