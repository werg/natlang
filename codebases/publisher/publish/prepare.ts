export default async function prepare(document: Document, target: string): Promise<PublishReport> {
return await host.publisher.publish(document, target);
}
