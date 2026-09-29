/** Authority instructions remain first. Program guidance precedes the call objective. */
export function programGuidance(text: string): string {
  return text ? '\n\n<natlang_program_guidance>\n' + text + '\n</natlang_program_guidance>\n' : '';
}
