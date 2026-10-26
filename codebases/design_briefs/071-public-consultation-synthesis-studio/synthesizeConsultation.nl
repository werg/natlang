---
description: Synthesize consultation submissions without collapsing mixed views, participant volume or minority concerns.
args: { docket: ConsultationDocket; delta: SubmissionDelta; update: ConsultationUpdate; editor: SynthesisEditor }
returns: SynthesisProposal
---
Keep each submission as an independently attributable unit. Include a late submission only when its exception authorizes it. Repeated campaign language explains a relation; it does not invalidate participant identity or independent views.

Bind scopedArguments = `extractScopedArguments(docket, delta)`. Preserve every position and reason in each submission, including conditional support and mixed views. Bind factual claims and proposed actions to original context; do not invent motive.

Bind campaigns = `analyzeCampaignRelations(docket, scopedArguments)`. Separate duplicate ingestion from multiple participants using shared template language. Keep unique contributions and participant counts separate from template-cluster frequency.

Bind themes = `mapConsultationThemes(docket, scopedArguments, campaigns)`. Group by underlying concern and reasoning, preserve minority and cross-cutting positions, and label disagreement factual, value-based, mixed or unknown only when the record supports it.

Bind responses = `draftPublicResponses(docket, themes, update)`. Present clarification, investigation and policy alternatives to the accountable public body without choosing its priorities or claiming consensus.

Bind frozen = `auditAndFreezeSynthesis(docket, delta, update, scopedArguments, campaigns, themes, responses)`. Trace every argument to a theme or explicit disposition, reconcile exact denominators independently from campaign clusters, anonymize private examples unless consent permits disclosure, and record challenged-omission revisions.

Return `SynthesisProposal` for `editor.id` with the versioned report and a resumable coverage graph. Ready means complete for public-body review; the report is not thereby approved or published.
