/** Shared by the executing host and the authenticated trajectory reader. */
const RESULT_PREFIX = ' as the result. If this is the result of the task you were given and you are satisfied with it, call return_result with status "success" and omit value to finish using this exact stored result. ';
export const STAGED_RESULT_NOTICE = RESULT_PREFIX + 'A JavaScript `return` in eval only stages this value; it does not end this invocation. To complete a child step or iteration transition, finish it now so its caller receives this state and continues. You can keep working and return a different value later only when more work belongs in this same invocation.';

const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const noticePattern = new RegExp('(?:^|\\n)Staged ([\\s\\S]+)' + escapeRegex(STAGED_RESULT_NOTICE) + '$');

/** Require a complete final host notice; a console imitation followed by normal
 * eval output is not a staged result. Action/invocation custody is checked by the caller. */
export function stagedResultBody(output: string): string | undefined {
  return noticePattern.exec(output)?.[1];
}
