# BitCliff AWS budget

Canon for the aws-ops skill (`SKILL.MD`). Deviations require explicit user
approval.

| line | amount |
|---|---|
| Allocation (AWS credits) | $5,000 |
| Project cap (confirmatory GPU work to date) | $250 |
| Spent, as of 2026-10-14 | ≈ $130.5 (0B ≈ $98, 0b2 ≈ $2.10, 0B-prime ≈ $30 including the $1.93 earlier attempts; OPEN_QUESTIONS §18) |
| 0B-prime cap | $50 hard stop, benchmark included (set by user 2026-09-27) |

## Burn-report rule

Every capped run reports burn to the user at 50%, 80%, and 100% of its cap.
At 100% the run stops. For 0B-prime that means reports at $25, $40, and $50.

Burn reports come from a timer on the box, not from the session's attention.
The timer fires at every hour of runtime and at each cap threshold. An idle
watchdog on the box terminates it after 30 minutes with no pipeline process.
Both are armed before any download starts (`SKILL.MD`, rule added 2026-10-14).

Standing thresholds from `SKILL.MD` still apply: stop and ask before any
action over $500 in credits or any cash spend over $50.
