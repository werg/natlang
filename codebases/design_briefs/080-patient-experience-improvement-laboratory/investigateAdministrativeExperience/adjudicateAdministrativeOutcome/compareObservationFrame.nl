---
args: {"trial": "AdministrativeTrial", "metrics": "AdministrativeMetric[]"}
returns: "string[]"
---
Compare metric definitions, populations, periods, numerators, denominators, available channels and workload definitions before calculating any difference. Call detectChannelAvailabilityShift(metrics) and retain its findings. A closed reporting channel can reduce complaint/call counts while access worsens. Ratios with zero, absent or changed denominators cannot establish improvement; keep actual metric artifact IDs and exact counts in the comparison.
