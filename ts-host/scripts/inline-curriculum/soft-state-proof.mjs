/** Validate a real Neuralese writer->reader relation from exact runtime graph events. */
export function validateExpectedReadCount({ actualCount, expectedPaths, childCallId }) {
  if (!Number.isInteger(actualCount) || !Array.isArray(expectedPaths))
    throw new Error('source-read counts need an integer actualCount and an expected path list');
  if (actualCount !== expectedPaths.length)
    throw new Error(`child ${childCallId} has ${actualCount} source reads, expected ${expectedPaths.length}`);
  return { expected_count: expectedPaths.length, actual_count: actualCount,
    no_source_read_expected: expectedPaths.length === 0 };
}

/** Resolve the writer's declared port against the immediate consumer's authored input declaration. */
export function resolveSoftStateArgument({ producerNextArgument, consumerExpectedArgument, fallback = 'prior' }) {
  if (consumerExpectedArgument && producerNextArgument && consumerExpectedArgument !== producerNextArgument)
    throw new Error(`producer declares next argument ${producerNextArgument}, but consumer declares ${consumerExpectedArgument}`);
  return consumerExpectedArgument ?? producerNextArgument ?? fallback;
}

/** Count observed tool reads separately from complete files supplied through FileHandle openings. */
export function summarizeSourceEvidence(proofCases) {
  if (!Array.isArray(proofCases)) throw new Error('source evidence summary requires a proof-case list');
  return proofCases.reduce((summary, proofCase) => {
    for (const child of proofCase.clean_child_reads ?? []) {
      summary.successful_source_reads += Array.isArray(child.source_reads) ? child.source_reads.length : 0;
      summary.complete_filehandle_openings += child.exact_complete_filehandle_opening === true ? 1 : 0;
    }
    return summary;
  }, { successful_source_reads: 0, complete_filehandle_openings: 0 });
}

/** Match a complete parameter declaration, rejecting unions and substring lookalikes. */
export function signatureHasExactParameter(signatureValue, argument, type) {
  const signature = String(signatureValue ?? '');
  const escapeRegex = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const exactParameter = new RegExp(`(?:^|\\(|, )${escapeRegex(argument)}: ${escapeRegex(type)}(?=, |\\))`);
  return exactParameter.test(signature);
}

export function validateSoftStateEdge({ graph, actualValue, expectedType = 'Neuralese<string>', writerCallId,
  consumerCallId, consumerArgument, writerNode }) {
  if (!actualValue || typeof actualValue !== 'object' || actualValue.$neuralese?.type !== expectedType ||
      typeof actualValue.$neuralese?.id !== 'string')
    throw new Error(`writer ${writerCallId} did not return ${expectedType}`);
  const block = actualValue.$neuralese.id;
  const writer = graph.find(event => event.kind === 'block_write' && event.call_id === writerCallId && event.block === block &&
    (writerNode === undefined || event.node === writerNode));
  if (!writer || typeof writer.node !== 'string') throw new Error(`no actual block_write for ${block} from ${writerCallId}`);
  const invocation = graph.find(event => event.kind === 'invocation' && event.call_id === consumerCallId && event.phase === 'start');
  const invocationEdge = invocation?.inputs?.find(input => input.node === writer.node && input.block === block &&
    input.port === `arg:${consumerArgument}`);
  if (!invocationEdge) throw new Error(`no exact ${consumerArgument} invocation edge from ${writer.node} to ${consumerCallId}`);
  const signature = String(invocation.signature ?? '');
  if (!signatureHasExactParameter(signature, consumerArgument, expectedType))
    throw new Error(`consumer ${consumerCallId} does not declare ${consumerArgument}: ${expectedType} (${signature})`);
  const read = graph.find(event => event.kind === 'block_read' && event.call_id === consumerCallId && event.block === block &&
    event.inputs?.some(input => input.node === writer.node && input.block === block && input.port === 'block'));
  if (!read) throw new Error(`no block_read from ${writer.node} to ${consumerCallId} for ${block}`);
  return { block, writer_call_id: writerCallId, writer_node: writer.node, consumer_call_id: consumerCallId,
    consumer_signature: signature, invocation_input_port: invocationEdge.port, block_read_node: read.node,
    exact_runtime_writer_to_reader_link: true };
}
