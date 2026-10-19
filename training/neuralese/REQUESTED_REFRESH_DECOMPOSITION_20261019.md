# Requested adaptive continuation

The supervisor's natural-language production policy decides when a useful
change warrants interrupting a training quantum. The following crisp mechanism
implements that decision without changing the objective or counting an
incomplete recipe as complete.

1. After verifying the actual trainer PID, birth and sandbox boot, the supervisor
   sends SIGUSR2 to request a semantic refresh. SIGTERM, SIGINT and SIGUSR1 keep
   their existing operational-stop behavior.
2. The shared native trainer records the request using the existing StopSignal
   mechanism. It finishes the current successful update and exposure accounting,
   saves optimizer, RNG and cursor, and enters the existing refresh qualification
   path. It does not wait for the declared relative quantum.
3. The existing continuation receipt reports the earned updates and actual
   runtime qualification. The request grants neither qualification nor data
   exposure. The successor consumes that genuine receipt through the shared
   supervisor and continues all remaining production obligations.

This applies to both students and to the preference trainer's shared native
training loop. Existing frozen runtimes require their explicitly recorded saved
update recovery; this change does not retroactively change their signal meaning.
