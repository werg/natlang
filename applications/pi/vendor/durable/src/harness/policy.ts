/**
 * PATCH (natlang port): pi-durable's scheduler policy restated over facts (`SchedulerPolicy`), and the facts' ownership
 * walks. The scheduler's inline code stays the default; this implementation drives the policy path with pi's own rules,
 * so the policy plumbing can be checked against pi-durable's whole suite, and it is the reference the natural-language
 * implementation is compared with.
 */
import type {
	AbortConversationDecision,
	AbortConversationFacts,
	AbortTaskDecision,
	AbortTaskFacts,
	BlockedReason,
	OwnerStep,
	PassDecision,
	PassFacts,
	ReconcileDecision,
	ReconcileFacts,
	SchedulerPolicy,
	StepDecision,
	StepFacts,
	TaskFact,
} from "./types.ts";

/**
 * Every task with live ordinary owned work (spec §5.5): each live non-background task counts for every owner task met
 * walking up from its parent, up to and including the first background owner; an unknown step ends the walk.
 */
export function ownedLiveOf(tasks: readonly TaskFact[]): Map<number, number[]> {
	const owned = new Map<number, number[]>();
	for (const task of tasks) {
		if (task.background) continue;
		for (const step of task.above) {
			if (step.kind === "unknown") break;
			if (step.kind !== "task") continue;
			const below = owned.get(step.id!);
			if (below === undefined) owned.set(step.id!, [task.id]);
			else below.push(task.id);
			if (step.background) break;
		}
	}
	return owned;
}

/**
 * Whether ordinary traversal from `scope` reaches the walk's start: walking up reaches the scope's conversation (or an
 * ownerless top for `null`) without crossing a background owner, unless `crossBackground`. `undefined` at an unknown step.
 */
export function inScopeOf(above: readonly OwnerStep[], scope: number | null, crossBackground = false): boolean | undefined {
	for (const step of above) {
		if (step.kind === "unknown") return undefined;
		if (step.kind === "conversation") {
			if (scope !== null && step.id === scope) return true;
		} else if (step.background && !crossBackground) {
			return false;
		}
	}
	return scope === null;
}

/** A live owner's cancellation intent reaches the walk's start before a background owner without it. */
export function belowCancelledOf(above: readonly OwnerStep[]): boolean {
	for (const step of above) {
		if (step.kind === "unknown") return false;
		if (step.kind !== "task") continue;
		if (step.status !== "terminal" && (step.abortRequested || (step.status === "completing" && step.outcome !== "completed"))) {
			return true;
		}
		if (step.background) return false;
	}
	return false;
}

const isFailed = (status: string | undefined, outcome: string | undefined): boolean =>
	(status === "completing" || status === "terminal") && outcome !== undefined && outcome !== "completed";

function fit(task: TaskFact, facts: PassFacts): { reason: BlockedReason } | { migrate: boolean } {
	const definition = facts.definitions.find((item) => item.kind === task.kind);
	if (definition === undefined) return { reason: "missing_task" };
	if (definition.version === task.version) return { migrate: false };
	if (definition.version < task.version) return { reason: "task_too_old" };
	if (facts.migrationFailed.includes(task.id)) return { reason: "migration_failed" };
	return { migrate: true };
}

export const crispSchedulerPolicy: SchedulerPolicy = {
	pass(facts: PassFacts): PassDecision {
		const owned = ownedLiveOf(facts.tasks);
		const live = new Set(facts.tasks.map((task) => task.id));
		const reserve: { id: number; mode: "run" | "abort"; migrate: boolean }[] = [];
		const orphan: { id: number; reason: BlockedReason }[] = [];
		for (const task of facts.tasks) {
			if (task.invocation !== null) continue;
			const waitingOn = task.abortRequested
				? (owned.get(task.id) ?? [])
				: task.status === "waiting"
					? (task.on ?? []).filter((id) => live.has(id))
					: [];
			if (waitingOn.length > 0 || task.status === "completing") continue;
			const mode = task.abortRequested ? "abort" : "run";
			const resolved = fit(task, facts);
			if ("reason" in resolved) {
				if (mode === "abort") orphan.push({ id: task.id, reason: resolved.reason });
				continue;
			}
			reserve.push({ id: task.id, mode, migrate: resolved.migrate });
		}
		const idle = facts.idleScopes.map((scope) => ({
			scope,
			idle: !facts.tasks.some((task) => !task.background && inScopeOf(task.above, scope) !== false),
		}));
		return { reserve, orphan, idle };
	},

	step(facts: StepFacts): StepDecision {
		if (facts.task === null || facts.task.status !== "running" || facts.closing) return { kind: "stop" };
		if (facts.mode === "abort") {
			return {
				kind: "fault",
				message: facts.previous?.error ?? `Abort handler of task ${facts.task.id} returned without a terminal outcome`,
			};
		}
		if (facts.task.abortRequested) return { kind: "stop" };
		if (facts.previous === null) return { kind: "continue" };
		if (facts.previous.error !== undefined) return { kind: "fault", message: facts.previous.error };
		if (facts.previous.unchanged) {
			return {
				kind: "fault",
				message: `Task ${facts.task.kind} phase ${facts.previous.phase} returned without durable progress`,
			};
		}
		if (!facts.definition.same) {
			if (facts.definition.exists && facts.definition.canReserve) return { kind: "handover" };
			if (!facts.reported) {
				return { kind: "continue", report: `Task ${facts.task.id} keeps running under its old ${facts.task.kind} definition` };
			}
		}
		return { kind: "continue" };
	},

	reconcile(facts: ReconcileFacts): ReconcileDecision {
		const mark: number[] = [];
		const add = (id: number, abortRequested: boolean): void => {
			if (!abortRequested && !mark.includes(id)) mark.push(id);
		};
		for (const task of facts.tasks) if (!task.background && belowCancelledOf(task.above)) add(task.id, task.abortRequested);
		for (const waiter of facts.failFast) {
			if (!waiter.members.some((member) => isFailed(member.status, member.outcome))) continue;
			for (const member of waiter.members) {
				if (member.live && !isFailed(member.status, member.outcome)) add(member.id, member.abortRequested);
			}
		}
		const withdraw = facts.queued.filter((queued) => belowCancelledOf(queued.above)).map((queued) => queued.conversation);
		const finalize: { id: number }[] = [];
		let remaining = [...facts.tasks];
		for (let round = 0; round <= facts.tasks.length; round++) {
			const owned = ownedLiveOf(remaining);
			const done = remaining.filter((task) => task.status === "completing" && !owned.has(task.id));
			if (done.length === 0) break;
			for (const task of done) finalize.push({ id: task.id });
			remaining = remaining.filter((task) => !done.includes(task));
		}
		return { mark, withdraw, finalize };
	},

	abortTask(facts: AbortTaskFacts): AbortTaskDecision {
		const task = facts.task;
		if (task === null) return { kind: "reject", message: `Task ${facts.id} does not exist`, join: false };
		if (task.status === "terminal") return { kind: "terminal", join: false };
		if (facts.invocation === null && task.status !== "completing" && !ownedLiveOf(facts.tasks).has(task.id)) {
			if (facts.blocked !== null) return { kind: "orphan", reason: facts.blocked, join: false };
		}
		return { kind: "mark", join: facts.invocation === "run" };
	},

	abortConversation(facts: AbortConversationFacts): AbortConversationDecision {
		const reached = facts.tasks.filter(
			(task) =>
				(facts.background || !task.background) && inScopeOf(task.above, facts.conversation, facts.background) === true,
		);
		return {
			mark: reached.filter((task) => !task.abortRequested).map((task) => task.id),
			withdraw: facts.queued
				.filter((queued) => inScopeOf(queued.above, facts.conversation, facts.background) === true)
				.map((queued) => queued.conversation),
			waitFor: facts.background ? reached.map((task) => task.id) : [],
		};
	},
};
