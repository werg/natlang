// PATCH (natlang port): select the policy path for every Harness the suite opens (see vitest.policy.config.ts).
import { Harness } from "../src/index.ts";
import { crispSchedulerPolicy } from "../src/harness/policy.ts";
import { admitSubmission } from "../src/harness/submissions.ts";
import type { AdmissionPolicy } from "../src/harness/types.ts";

const crispAdmission: AdmissionPolicy = {
	admit: (request, context) =>
		request.session.commit(
			(tx) => admitSubmission(tx, request.conversationId, request.draft, request.now, request.queueModes),
			context,
		),
};
const open = Harness.open;
(Harness as { open: typeof open }).open = (storage, options, context) =>
	open(storage, { schedulerPolicy: crispSchedulerPolicy, admission: crispAdmission, ...options }, context);
