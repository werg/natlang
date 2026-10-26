---
description: Link customer events, order failures and dependent promises across the complete case timeline.
args:{cases:RecoveryCase[];timeline:EvidencePage[]}
returns:RecoveryIssueEpisode[]
---
Order interactions by source time and retain the original promise wording and evidence references. Group failures by affected order and customer outcome. Mark independent only when their resolution does not change another issue's remedy or timing value. Keep uncertain chronology as unresolved rather than imposing an order.
