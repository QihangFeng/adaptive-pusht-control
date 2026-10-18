# Stopped dataset preflight

The original frozen protocol is preserved here. No training update or new policy
prediction was made under it.

All 31 recovery continuations and 20 diagnostic baseline episodes replayed with
exact saved RGB/state agreement. Preparation then stopped at expert episode 0:
the public 161-row demonstration marks `next.done` true at rows 159 and 160,
whereas the initial validator required it only in the final row. Done was not a
success label (`next.success` was false).

The prepared local replay cache was preserved in the Git-ignored directory
`data/prepared/robotics/recovery-pilot-preflight-terminal-check/`. It is not
silently rebound to a different protocol. The revised preparation explicitly
handles public terminal markers and rebuilds the cache under its new hashes.
The training budget, scene split and sixty-attempt comparison remain unchanged.
