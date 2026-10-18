# Bounded provider metadata reads

The scheduling policy remains with the existing native controller. The exact
transport should not leave a GPU idle merely to discover whether a tiny request
or stop file exists. Existing provider timings show several seconds per remote
call; optional reads currently require existence, size and content operations.

1. Validate the same allowed lexical metadata path before any remote call.
2. In one remote operation, distinguish an absent optional file from an empty
   file, open the file, check its size against the existing16MiB limit, and read
   at most that limit plus one byte for inline content. Reject a changed size or
   excessive content. Above64KiB, return only the checked size and retain the
   qualified SDK file read, validating its returned type and byte count. The
   installed SDK explicitly documents exec-stream truncation; do not send a
   sixteen-megabyte metadata file through stdout merely to save one call.
3. Return the exact bytes as base64 with a byte count. Verify the bounded envelope
   and decoded byte count on the host. Small reads use one call; larger reads
   use two and still avoid the former separate existence call. Keep request/content hash
   verification in consumers, private error evidence and SDK timing records.
4. Reuse this mechanism for required and optional reads. Do not change learner
   commands, ownership, model-facing text, numeric code, artifact transfer or
   admission. Adopt a new provider at a safe ownership boundary; stopping the
   current provider can stop its learner and is not a transport optimization.

Crisp code is justified by byte-exact external transport and bounds. Checks cover
actual binary/empty/missing files, directory behavior, namespace escape,16MiB
boundaries, malformed/truncated responses and one remote call with no SDK file
read for small metadata. Independent review precedes deployment; local tests do not establish
measured cloud latency improvement.
