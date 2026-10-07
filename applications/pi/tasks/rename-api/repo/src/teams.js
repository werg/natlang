import { allIds, getUser } from './users.js';
export async function team(name) {
  const users = await Promise.all((await allIds()).map(id => getUser(id)));
  return users.filter(user => user.team === name).map(user => user.name);
}
