/*---
engine: typescript-host
args:
  block_id: string
  input: string
returns: CellResult
---*/
return await host.wiki.runCell(args.block_id, args.input);
