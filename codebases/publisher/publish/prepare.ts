/*---
engine: typescript-host
args:
  document: Document
  target: Text
returns: PublishReport
---*/
return await host.publisher.publish(args.document, args.target);
