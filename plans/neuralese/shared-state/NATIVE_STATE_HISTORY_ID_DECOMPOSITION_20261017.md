# Native state references in tool history

The executing model may copy a shortened native state ID into a tool argument.
That malformed value must remain ordinary argument data for the normal typed
runtime error and recovery. It must not become a transport dependency that
prevents the next model turn. A complete valid native state ID retains its
existing restore and provenance requirements.

1. Identify actual native state references with the existing canonical ID
   predicate, in nested argument lists as well as standalone parts.
2. Preserve malformed lookalikes without inventing a state or matching an ID
   prefix. Ordinary tool validation decides whether the value fits its type.
   A content-part list containing a malformed native reference stays ordinary
   as a whole, preserving its exact JSON value shape in both transports.
3. Restore actual complete references through the existing store mechanism.
   Missing genuine artifacts still require recovery; never silently drop them.

These are natural-language transport requirements. Crisp code only enumerates
the exact dependencies using the existing predicate. Regression coverage uses
the actual truncated ID from the protected evaluation, full valid IDs, nested
JSON-encoded arguments, and mixed valid/invalid parts. The Python renderer must
use the same distinction when materializing argument values.
