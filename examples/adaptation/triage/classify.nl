---
args: { message: string }
returns: "'urgent' | 'routine' | 'unknown'"
---
Classify message. A security incident or immediate outage is urgent. Ordinary scheduling and informational requests are routine. When information is insufficient, return unknown.
