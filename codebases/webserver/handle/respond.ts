export default function respond(id: string, response: Response): boolean {
fx.http.respond(id, response)
return true
}
