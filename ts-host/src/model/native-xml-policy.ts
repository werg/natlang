/** SHA-256 of the exact native-xml-calls.ts parser implementation. */
export const NATIVE_XML_POLICY_SHA256 = '7729017bde453a3944fa8d774e472a8480d18b2548247725b5c676ace88bd5c5';
/**
 * Earlier parser policies whose captures the current parser reads identically. 1b2713ad…: before 2026-11-23 it refused
 * every schema holding prefixItems (tuple results); c93dc767…: before 2026-11-26 it ignored const/enum in types().
 */
export const NATIVE_XML_POLICY_READS = new Set([NATIVE_XML_POLICY_SHA256,
  'c93dc767e0bb16a415c022ea64b14e93f992f53d88750b5bdcc8dbcee765651f',
  '1b2713adb032003d3522794fe65b75f03d9b31a1453bc86027f544f460bc315e']);
