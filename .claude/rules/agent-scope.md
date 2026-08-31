---
name: Agent Scope
description: Rules for the scope of an agent's actions
paths:
  - "**/*.py"
---
Never modify a file that is outside your assigned task's explicit
scope, even if you believe it needs improvement. If you notice an
issue in a file you weren't asked to touch, report it, do not fix it
unprompted.

Before any multi-step or long-running task that touches shared code
(especially merge, pipeline, or data-processing scripts), state which
files you will modify at the start, and do not deviate from that list.

After any task that modifies code, report a summary of exactly what
changed, file by file. If a file was modified that wasn't part of the
stated scope, flag this explicitly and do not treat it as routine.

When running parallel or forked sub-tasks, each sub-task must only
write to the specific output files it was assigned (e.g. its own
batch CSV). No sub-task may modify a shared script, pipeline file, or
another sub-task's output.