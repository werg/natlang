export default async function inspect(request: Request, plan: Plan, receipt: Receipt): Promise<Inspection> {
return await host.media.inspect(request, plan, receipt);
}
