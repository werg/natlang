/** Text that is actually native tool syntax must never be accepted as a plain answer. */
export const TOOL_MARKUP = /<tool_call>|<\/tool_call>|<function=|<\/function>|<parameter=|<\/parameter>|<\|tool_call_start\|>|<\|tool_call_end\|>/;
