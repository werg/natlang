/** The platform's natlang roots for data, configuration, state and cache; each can be overridden by an environment variable. */
import { homedir, platform } from 'node:os';
import { join, resolve } from 'node:path';

export function defaultNatlangDataDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_HOME) return resolve(environment.NATLANG_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang');
  return join(environment.XDG_DATA_HOME ?? join(homedir(), '.local', 'share'), 'natlang');
}
export function defaultNatlangConfigDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_CONFIG_HOME) return resolve(environment.NATLANG_CONFIG_HOME);
  if (platform() === 'win32') return join(environment.APPDATA ?? homedir(), 'natlang');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang');
  return join(environment.XDG_CONFIG_HOME ?? join(homedir(), '.config'), 'natlang');
}
export function defaultNatlangStateDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_STATE_HOME) return resolve(environment.NATLANG_STATE_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang', 'state');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang', 'state');
  return join(environment.XDG_STATE_HOME ?? join(homedir(), '.local', 'state'), 'natlang');
}
export function defaultNatlangCacheDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_CACHE_HOME) return resolve(environment.NATLANG_CACHE_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang', 'cache');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Caches', 'natlang');
  return join(environment.XDG_CACHE_HOME ?? join(homedir(), '.cache'), 'natlang');
}
