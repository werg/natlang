/*---
args:
  commands: Command[]
returns: number
effects:
- queue.send
---*/
for (const command of args.commands) fx.queue.send(command);
return args.commands.length;
