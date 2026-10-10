/**
 * pi's system prompt on the Node host: context files and skills loaded from the file system as pi loads them
 * (host/resources.ts), and pi's package directory from $PI_PACKAGE_DIR, else this application's directory.
 */
import { fileURLToPath } from 'node:url';
import type { Extension } from '../../vendor/durable/src/harness/types.ts';
import { Resources } from '../../host/resources.ts';
import { piPrompt } from './sections.ts';

const APP_DIR = fileURLToPath(new URL('../..', import.meta.url));

export type NodePiPromptOptions = {
  /** The directory used when neither the environment nor the agent names one. */
  cwd: string;
  /** pi's package directory for the docs section: default $PI_PACKAGE_DIR, else this application's directory. */
  packageDir?: string;
  /** Extra skill paths (pi's settings `skills`). */
  skillPaths?: string[];
  /** pi's agent directory; default $PI_CODING_AGENT_DIR or ~/.pi/agent. */
  agentDir?: string;
};

/** pi's system prompt with the Node host's resources. */
export function nodePiPrompt(options: NodePiPromptOptions): Extension {
  return piPrompt({ cwd: options.cwd, packageDir: options.packageDir ?? process.env.PI_PACKAGE_DIR ?? APP_DIR,
    resources: new Resources({ ...(options.agentDir ? { agentDir: options.agentDir } : {}), skillPaths: options.skillPaths ?? [] }) });
}
