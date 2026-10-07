const table = new Map([[1, { id: 1, name: 'Ada', team: 'core' }], [2, { id: 2, name: 'Grace', team: 'tools' }], [3, { id: 3, name: 'Linus', team: 'core' }]]);
/** The user with this id, or null. */
export async function getUser(id) { return table.get(id) ?? null; }
export async function allIds() { return [...table.keys()]; }
