"""The digest operator's write site in training (DECISIONS.md 43), mirroring ts-host/src/neuralese/digest.ts.

The runtime writes a digest of a large argument at this write site and lists the block, with `digest_note`, in place of
the cut-off preview. Training builds the same site from a converted record's digest part, so what is trained is what
runs; tests/fixtures/digest-site.json pins the two implementations to the same text.
"""

from __future__ import annotations

PREFIX = "const digest: Neuralese<Digest> = "
SOURCE_CHARS = 48_000


def digest_site(system, name: str, type: str, value: str, instructions: str) -> list[dict]:
    """The write site's messages. `system` is the digest instructions: their text, or parts with their soft form."""
    if len(value) > SOURCE_CHARS:
        value = value[:SOURCE_CHARS] + f" <<cut off: {len(value) - SOURCE_CHARS} of {len(value)} characters not shown>>"
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"The call that receives the value has these instructions:\n{instructions}\n\n"
                                        f"The value of {name} ({type}):\n{value}"}]


def digest_note(holder: str) -> str:
    return f"  // digest of the value; {holder} holds all of it"
