/*---
engine: typescript-host
args:
  document: Document
  target: string
returns: PublishReport
---*/
return await host.publisher.publish(args.document, args.target);
