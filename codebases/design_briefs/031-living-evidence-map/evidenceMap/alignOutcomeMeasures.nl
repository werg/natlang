---
args: { observations: ObservationLedger, conversions: EvidencePage[] }
returns: MeasurementAlignment
---
Align outcomes only where construct, instrument, timepoint and population are comparable. Preserve original units and require an approved conversion artifact before comparing different units; without it, mark measures unconverted or incomparable rather than normalize by guesswork.
