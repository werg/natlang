export default async function validate(revision: string): Promise<Validation> {
return await host.repository.validate(revision);
}
