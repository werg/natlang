import writing from './writing.mjs';
import worlds from './worlds.mjs';
import operations from './operations.mjs';
import workbenches from './workbenches.mjs';
export const apps = [...writing, ...worlds, ...operations, ...workbenches];
export const appById = new Map(apps.map(app => [app.id, app]));
