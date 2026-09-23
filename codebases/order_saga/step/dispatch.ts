import { queue } from 'natlang:services';
export default function dispatch(commands: Command[]): number {
for (const command of commands) queue.send(command);
return commands.length;
}
