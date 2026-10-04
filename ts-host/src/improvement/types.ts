import type { FolderSnapshot } from '../native/scoped-fs.js';
import type { SkillUseEvent } from '../skills/observability.js';
export type ImprovementCase = { id: string; group: string; split: 'train' | 'validation' | 'test'; args: unknown[]; expected: unknown; services?: Record<string, string>; folder?: Record<string, string>; expectedFiles?:Record<string,string>;files?:Record<string,string>; required?:boolean };
export type ProgramContract = { entry: string; exportName: string; signature?: string; programId: string };
export type CheckReport = { source: string; valid: boolean; diagnostics: string[]; contract: ProgramContract };
export type Outcome = { caseId: string; passed: boolean; quality: number; gates: Record<string,boolean>; value?: unknown; modelCalls?:number; modelTrace?:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}[]; skillUseTrace?:SkillUseEvent[]; modelTraceTruncated?:boolean; serviceDeclarations?:Record<string,string>; failureKind?:'fixture'|'target'|'timeout'; error?: string; evidence: string; args?:unknown[];expected?:unknown;expectedFiles?:Record<string,string> };
export type EvaluationReport = { source: string; split: 'train' | 'validation' | 'test'; suiteVersion: string; quality: number; modelCalls?:number; gatesPassed: boolean; passed: number; total: number; evidence: string; sourceBytes?:number; scores?: { caseId: string; quality: number }[]; outcomes?: Outcome[] };
export type TransformationSpec = { id: string; intent: string; preserves: string[]; changes: string[]; checks: string[]; allowedFiles?: string[]; mode?: 'instruction'|'structural' };
export type TransformationObligation = { obligation:string; status:'verified'|'violated'|'empirically-supported'|'unverified'; method:string; evidence:string[] };
export type TransformationResult = { status: 'verified' | 'violated' | 'empirically-supported' | 'unverified'; source: string; spec: TransformationSpec; evidence: string[]; reason: string; obligations?: TransformationObligation[]; eligible?: boolean };
export interface ProgramEvaluator {
  check(folder: FolderSnapshot): Promise<CheckReport>;
  evaluate(folder: FolderSnapshot, request: { split: 'train' | 'validation'; caseIds?: string[]; seed?: number }): Promise<EvaluationReport>;
  page(evidence: string, start?: number, limit?: number): Outcome[];
}
