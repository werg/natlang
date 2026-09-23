export default function dispatch(commands: Command[]): number {
for (const command of commands) fx.queue.send(command);
return commands.length;
}
