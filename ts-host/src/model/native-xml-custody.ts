/** Reauthenticate the exact sampled assistant-content call, without relabelling it as a provider raw_call. */
import { hexDigest } from '../native/hash.js';
import { parseNativeXmlCalls } from './native-xml-calls.js';
import { NATIVE_XML_POLICY_SHA256 } from './native-xml-policy.js';
type Dict = Record<string, unknown>;
const obj=(v:unknown): Dict => { if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('native XML custody needs an object'); return v as Dict; };
const canonical=(v:unknown):string => Array.isArray(v) ? `[${v.map(canonical).join(',')}]` : v && typeof v==='object' ? `{${Object.entries(v as Dict).sort(([a],[b])=>a.localeCompare(b)).map(([k,x])=>`${JSON.stringify(k)}:${canonical(x)}`).join(',')}}` : JSON.stringify(v);
const same=(a:unknown,b:unknown) => canonical(a) === canonical(b);
export function authenticateNativeXmlTurn(turn: Dict): [string, Dict][] {
  const response=obj(turn.model_response), proof=obj(response.tool_parse), request=obj(turn.model_request), raw=obj(response.raw_response);
  if (proof.schema !== 'natlang.native-xml-tool-parse/1' || proof.source !== 'assistant-content' ||
      proof.schema_custody !== 'response_tools' || proof.policy_sha256 !== NATIVE_XML_POLICY_SHA256)
    throw new Error('native XML parser policy differs');
  if (typeof turn.model_request_canonical !== 'string' || typeof response.raw_response_canonical !== 'string' ||
      hexDigest(turn.model_request_canonical) !== turn.request_sha256 || hexDigest(response.raw_response_canonical) !== turn.raw_response_sha256)
    throw new Error('native XML exact request or response digest differs');
  const withoutInvocation={...request};delete withoutInvocation.invocation_id;
  if (!same(JSON.parse(turn.model_request_canonical),withoutInvocation) || !same(JSON.parse(response.raw_response_canonical),raw) ||
      request.invocation_id !== turn.invocation_id || !same(request.messages,turn.context) || !same(request.tools,turn.tools_offered) ||
      !Array.isArray(request.tools) || request.tools.length || !Array.isArray(request.response_tools) ||
      !same(request.response_tools,turn.response_tools) || !Array.isArray(response.raw_calls) || response.raw_calls.length)
    throw new Error('native XML request, schema or original raw-call custody differs');
  const aliases=obj(proof.wire_tool_aliases), names=new Set<string>();
  const tools=(request.response_tools as unknown[]).map(item=>{
    const tool=structuredClone(obj(item));const fn=obj(tool.function);const name=fn.name;
    if (tool.type!=='function' || typeof name!=='string') throw new Error('invalid allowed tool');
    const alias=aliases[name];if (alias!==undefined && typeof alias!=='string') throw new Error('invalid tool alias');
    fn.name=alias??name;if(names.has(fn.name as string))throw new Error('ambiguous wire tool alias');names.add(fn.name as string);
    return {function:fn as {name:string;parameters?:unknown}};
  });
  const choices=raw.choices;if(!Array.isArray(choices) || !choices.length)throw new Error('missing provider response choice');
  const choice=obj(choices[0]),message=obj(choice.message);
  if(choice.finish_reason==='length' || typeof message.content!=='string' || (Array.isArray(message.tool_calls) && message.tool_calls.length))throw new Error('native XML response incomplete or provider already sent calls');
  const reverse=Object.fromEntries(Object.entries(aliases).map(([a,b])=>[b,a]));
  const calls=parseNativeXmlCalls(message.content,tools).map(([name,args])=>[reverse[name]??name,args] as [string,Dict]);
  const assistant=obj(turn.assistant), recorded=assistant.calls;
  if (!Array.isArray(recorded) || recorded.length!==calls.length || response.text!=='' || assistant.content!=='' ||
      !same(response.calls,calls))throw new Error('native XML parsed action differs from recorded response');
  calls.forEach(([name,args],i)=>{const call=obj(recorded[i]);if(call.tool!==name || call.source_tool!==name || !same(call.arguments,args))throw new Error('native XML recorded call differs');});
  const reasoning=message.reasoning_content ?? message.reasoning;
  if ((typeof reasoning==='string' && reasoning.trim()?reasoning:null)!==(assistant.reasoning??null))throw new Error('native XML reasoning custody differs');
  return calls;
}
