/** Authority instructions remain first. Program guidance precedes the call objective. */
export type MemeticGuidanceBinding = {
  schema: 'natlang.memetic-program-guidance/1'; program_id: string;
  student: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string };
  text: string; native_state_id: string; native_state_artifact: { path: string; sha256: string; bytes: number };
  program_guidance: string;
};
export type NativeProgramGuidance = { readonly kind: 'native-program-guidance'; readonly text: string;
  readonly program_id?: string; readonly binding?: Readonly<MemeticGuidanceBinding> };

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
        b.program_guidance !== `${b.native_state_id}`)
      throw new TypeError('invalid natlang.memetic-program-guidance/1 binding');
    return Object.freeze({ kind: 'native-program-guidance', text: b.text, binding: structuredClone(b) });
  }
  const text = input;
  if (typeof text !== 'string' || !text.trim() || typeof programId !== 'string' || !programId)
    throw new TypeError('native program guidance text and program identity must be nonempty');
  return Object.freeze({ kind: 'native-program-guidance', text, program_id: programId });
}

export function programGuidance(guidance: string | NativeProgramGuidance): string {
  if (typeof guidance !== 'string') return '';
  const text = guidance;
  return text ? '\n\n<natlang_program_guidance>\n' + text + '\n</natlang_program_guidance>\n' : '';
}
