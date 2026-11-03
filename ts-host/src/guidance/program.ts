/** Authority instructions remain first. Program guidance precedes the call objective. */
export type MemeticGuidanceBinding = {
  schema: 'natlang.memetic-program-guidance/1'; program_id: string;
  student: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string };
  text: string; native_state_id: string; native_state_artifact: { path: string; sha256: string; bytes: number };
  program_guidance: string;
  /** Function scope: only calls of this function (the `.nl` file stem) import it; absent, every call of the program does.
   * A function binding's text is complete for its calls (program-wide part first, then the function's own part). */
  function?: string;
};
export type NativeProgramGuidance = { readonly kind: 'native-program-guidance'; readonly text: string;
  readonly program_id?: string; readonly binding?: Readonly<MemeticGuidanceBinding> };
/** Scoped guidance of one program: each call imports at most one binding, its function's if present, else the program-wide one. */
export type NativeProgramGuidanceSet = { readonly kind: 'native-program-guidance-set'; readonly program_id: string;
  readonly program?: NativeProgramGuidance; readonly functions: Readonly<Record<string, NativeProgramGuidance>>;
  readonly bindings: readonly Readonly<MemeticGuidanceBinding>[] };
export type ProgramGuidance = NativeProgramGuidance | NativeProgramGuidanceSet;
export type ProgramGuidanceInput = MemeticGuidanceBinding | readonly MemeticGuidanceBinding[];
const FUNCTION_NAME = /^[A-Za-z_$][\w$]*$/;
export type AuthoredTextGuidance = { readonly kind: 'authored-text-guidance'; readonly program_id: string; readonly text: string };

/** Bind stored guidance to the current native reader when a runtime call begins. */
export function nativeProgramGuidance(input: string | MemeticGuidanceBinding, programId?: string): NativeProgramGuidance {
  if (typeof input !== 'string') {
    const b = input;
    const identity = b?.student;
    if (b?.schema !== 'natlang.memetic-program-guidance/1' || typeof b.program_id !== 'string' || !b.program_id ||
        typeof b.text !== 'string' || !b.text.trim() || !identity ||
        typeof identity.architecture !== 'string' || !identity.architecture ||
        ![identity.checkpoint_sha256, identity.tokenizer_fingerprint, identity.embedding_weight_sha256]
          .every(value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value)) ||
        !/^nzs1_[a-z2-7]{52}$/.test(b.native_state_id) ||
        !b.native_state_artifact || typeof b.native_state_artifact.path !== 'string' || !b.native_state_artifact.path.startsWith('/') ||
        typeof b.native_state_artifact.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(b.native_state_artifact.sha256) ||
        !Number.isSafeInteger(b.native_state_artifact.bytes) || b.native_state_artifact.bytes < 1 ||
        b.program_guidance !== `${b.native_state_id}` ||
        (b.function !== undefined && (typeof b.function !== 'string' || !FUNCTION_NAME.test(b.function))))
      throw new TypeError('invalid natlang.memetic-program-guidance/1 binding');
    return Object.freeze({ kind: 'native-program-guidance', text: b.text, binding: structuredClone(b) });
  }
  const text = input;
  if (typeof text !== 'string' || !text.trim() || typeof programId !== 'string' || !programId)
    throw new TypeError('native program guidance text and program identity must be nonempty');
  return Object.freeze({ kind: 'native-program-guidance', text, program_id: programId });
}

/**
 * One binding, or a list of scoped bindings of one program: at most one program-wide binding (no `function`) and at
 * most one per function. A single program-wide binding stays the plain guidance it always was.
 */
export function programGuidance(input: ProgramGuidanceInput): ProgramGuidance {
  if (!Array.isArray(input)) {
    const single = nativeProgramGuidance(input as MemeticGuidanceBinding);
    return single.binding?.function === undefined ? single : programGuidance([input as MemeticGuidanceBinding]);
  }
  if (!input.length) throw new TypeError('scoped program guidance needs at least one binding');
  const guidance = input.map(binding => nativeProgramGuidance(binding));
  const programId = guidance[0]!.binding!.program_id;
  let program: NativeProgramGuidance | undefined;
  const functions: Record<string, NativeProgramGuidance> = {};
  for (const item of guidance) {
    const binding = item.binding!;
    if (binding.program_id !== programId) throw new TypeError('scoped program guidance bindings belong to different programs');
    if (binding.function === undefined) {
      if (program) throw new TypeError('scoped program guidance has more than one program-wide binding');
      program = item;
    } else {
      if (Object.hasOwn(functions, binding.function))
        throw new TypeError(`scoped program guidance has more than one binding for function ${binding.function}`);
      functions[binding.function] = item;
    }
  }
  return Object.freeze({ kind: 'native-program-guidance-set', program_id: programId, ...(program ? { program } : {}),
    functions: Object.freeze(functions), bindings: Object.freeze(guidance.map(item => item.binding!)) });
}

/** The program a guidance value belongs to. */
export function programGuidanceProgram(guidance: ProgramGuidance | undefined): string | undefined {
  return !guidance ? undefined : guidance.kind === 'native-program-guidance-set' ? guidance.program_id :
    guidance.binding?.program_id ?? guidance.program_id;
}

/** The exact bindings a run was given, for provenance digests (a single binding digests as itself). */
export function programGuidanceIdentity(guidance: ProgramGuidance): unknown {
  return guidance.kind === 'native-program-guidance-set' ? guidance.bindings : guidance.binding ?? guidance;
}

/** The one guidance a call of `functionName` imports: its function's binding, else the program-wide one, else none. */
export function guidanceForCall(guidance: ProgramGuidance | undefined, functionName: string | undefined): NativeProgramGuidance | undefined {
  if (!guidance) return undefined;
  if (guidance.kind === 'native-program-guidance-set')
    return (functionName && Object.hasOwn(guidance.functions, functionName) ? guidance.functions[functionName] : undefined) ?? guidance.program;
  const scope = guidance.binding?.function;
  return scope === undefined || scope === functionName ? guidance : undefined;
}

/** Text-only authored policy; separate from native state and bound to one program. */
export function authoredTextGuidance(text: string, programId: string): AuthoredTextGuidance {
  if (typeof text !== 'string' || !text.trim() || typeof programId !== 'string' || !programId)
    throw new TypeError('authored text guidance needs text and exact program identity');
  return Object.freeze({ kind: 'authored-text-guidance', program_id: programId, text });
}

export function renderAuthoredTextGuidance(guidance?: AuthoredTextGuidance): string {
  if (!guidance) return '';
  if (guidance.kind !== 'authored-text-guidance' || typeof guidance.program_id !== 'string' || !guidance.program_id ||
      typeof guidance.text !== 'string' || !guidance.text.trim())
    throw new TypeError('invalid authored text guidance');
  return '\n\n<natlang_program_guidance>\n' + guidance.text + '\n</natlang_program_guidance>\n';
}
