/*---
engine: typescript-host
args:
  block_id: Text
  input: Text
returns: CellResult
---*/
return await host.wiki.runCell(args.block_id, args.input);
