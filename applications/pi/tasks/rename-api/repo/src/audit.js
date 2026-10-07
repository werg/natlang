import * as users from './users.js';
/** Log line for an access; the user is looked up with users.getUser. */
export async function audit(id, action) {
  const who = await users.getUser(id);
  return `${new Date(0).toISOString()} ${who?.name ?? `#${id}`} ${action}`;
}
