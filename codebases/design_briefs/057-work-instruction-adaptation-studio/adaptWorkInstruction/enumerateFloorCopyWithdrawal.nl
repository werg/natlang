---
args: { copies: FloorCopy[], version: string }
returns: string[]
---
Find every non-withdrawn copy at a work location whose version is superseded by the proposed source version. Return a withdrawal task per copy with its location and accountable owner; publication does not itself establish withdrawal.
