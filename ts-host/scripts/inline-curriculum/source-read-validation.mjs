const SHOWN_CHARS = 2000;
function cutHeadLength(text) {
  let head = Math.floor(SHOWN_CHARS * 0.75);
  const headBreak = text.lastIndexOf('\n', head);
  if (headBreak > head - 200) head = headBreak;
  return head;
}
export function expectedSourcePageCount(text) {
  if (typeof text !== 'string') throw new Error('source text must be a string');
  if (text.length <= SHOWN_CHARS) return 0;
  return 1 + Math.ceil((text.length - cutHeadLength(text)) / SHOWN_CHARS);
}

/** Match a declared read_file call and any requested read_page segments to exact source bytes. */
export function matchDeclaredSourceRead({ readPath, expectedPath, resultText, folderFiles, pageEvents = [] }) {
  if (typeof readPath !== 'string' || typeof expectedPath !== 'string' || readPath !== expectedPath)
    return null;
  const sourceText = folderFiles?.[readPath];
  if (typeof sourceText !== 'string') return null;
  const pageCount = expectedSourcePageCount(sourceText);
  if (pageCount === 0) {
    if (resultText !== sourceText || pageEvents.length) return null;
    return { source_path: readPath, source_text: sourceText, exact_complete_source_read: true, page_count: 0 };
  }
  const pageId = /read_page\("([A-Za-z]+\d*)", 2\)/.exec(resultText ?? '')?.[1];
  if (!pageId || pageEvents.length !== pageCount) return null;
  const ordered = [...pageEvents].sort((a, b) => Number(a.arguments?.page) - Number(b.arguments?.page));
  if (ordered.some((event, index) => event.arguments?.id !== pageId || event.arguments?.page !== index + 1)) return null;
  const pieces = ordered.map((event, index) => {
    const suffix = index + 1 < pageCount
      ? new RegExp(`\\n<<page ${index + 1} of ${pageCount} shown; read_page\\(\\"${pageId}\\", ${index + 2}\\) shows the next part>>$`)
      : new RegExp(`\\n<<page ${pageCount} of ${pageCount}, the last>>$`);
    return String(event.result_text ?? '').replace(suffix, '');
  });
  if (ordered.some(event => !String(event.result_text ?? '').includes(`<<page ${event.arguments.page} of ${pageCount}`)) ||
      pieces.join('') !== sourceText) return null;
  return { source_path: readPath, source_text: sourceText, exact_complete_source_read: true,
    page_count: pageCount, page_id: pageId, page_numbers: ordered.map(event => event.arguments.page) };
}
