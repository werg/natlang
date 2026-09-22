/*---
engine: typescript-host
args:
  input: string
returns: Clip
---*/
return await host.media.probe(args.input);
