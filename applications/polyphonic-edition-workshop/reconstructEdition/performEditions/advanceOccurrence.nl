---
description: Own one occurrence boundary and consume each voice’s actual event-state reducer.
args:
  state: PerformanceFrontier
returns: PerformanceFrontier
---
1. If nextOccurrenceIndex equals occurrences.length return state. Select exactly that indexed actual
   occurrence and compute its absolute start from earlier source-declared section lengths.
2. Interpret protocol.resetPolicy for this occurrence. Initialize each actual voice's NotationState
   with its source-required accidental map, elapsed ticks and previous sound. In this closed protocol
   each occurrence begins fresh; an unsupported carry rule remains a source issue rather than a guess.
3. Gather this hypothesis's actual options for the occurrence section and voice, ordered by original
   event order. Initialize VoiceFrontier with those options, actual protocol/occurrence/start and empty
   local performed events. Do not modify the source-level chosen option on a reprise.
4. Call advanceVoiceEvent.iterateOn for each actual voice with measure pending.length. Consume each
   returned event list, final state and unresolved findings. Preserve previously completed voices if
   another voice has a source or execution failure.
5. Join only the actual returned local performances, append them to prior occurrences and advance
   nextOccurrenceIndex by one. Return the actual accumulated state. Ensemble checking owns cross-voice
   constraints; this boundary stage does not stretch a note to force synchronization.
