/** Content-free observation of skill availability and host-mediated use. A read does not imply cognitive use. */
export type SkillUsePhase = 'offered' | 'body_read' | 'support_file_read' | 'helper_invoked';
export type SkillUseEvent = {
  kind: 'skill_use';
  phase: SkillUsePhase;
  skill_name: string;
  skill_revision: string;
  invocation_id?: string | null;
  path?: string;
  helper_export?: string;
  interpretation?: 'listed_in_invocation_opening_not_awareness';
};
