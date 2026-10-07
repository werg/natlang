import { getUser } from './users.js';
export async function greet(id) {
  const user = await getUser(id);
  return user ? `Hello, ${user.name}!` : 'Hello, stranger!';
}
