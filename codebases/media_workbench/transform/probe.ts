/*---
engine: typescript-host
args:
  input: Text
returns: Clip
---*/
return await host.media.probe(args.input);
