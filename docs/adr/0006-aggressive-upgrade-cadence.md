# 6. Aggressive upgrade cadence

Status: accepted (2026-06-19)

## Context

The whole stack is bleeding edge: GoMLX tracks `main` (ADR-0001), go-xla/PJRT
churn, and the surrounding ecosystem (safetensors libs, mmap, HF protocol) moves.
Holding old versions to "stay stable" on a stack like this accrues more risk than
it avoids — you drift away from the only-maintained line and inherit migration
debt in one painful lump.

## Decision

Upgrade often, on purpose. The Go toolchain and **all** module dependencies are
bumped frequently and treated as routine maintenance, not events. This is an
accepted, named tax of building on a bleeding-edge foundation.

- Still pin for reproducibility: every module pins exact versions / commits in its
  `go.mod` (ADR-0001). Pinning and frequent bumping are not in tension — pin, then
  move the pin forward often.
- Bump per module (ADR-0005), so a breaking dep change is isolated and tested in
  one module before it spreads.
- Automate the bump rather than letting it drift: an automated dependency-update
  mechanism (e.g. a scheduled/event-driven bot opening per-module update PRs that
  run the full gate) — consistent with the project's event-driven/automated rules.
  A human reviews and merges; nobody hand-edits versions as the normal path.
- The backend boundary (ADR-0002) is what makes this affordable: vendor API breaks
  land in `backend/` only, so a GoMLX `main` bump rarely touches `model`/`train`.

## Consequences

- Recurring upgrade work is expected and budgeted, not a surprise. Green gate after
  a bump is the bar; a red bump is reverted or fixed, never merged.
- The per-module test gate and the backend-boundary grep test (ADR-0002) are the
  safety net that makes frequent bumps safe to merge.
- Occasional forced migrations (a GoMLX `main` refactor, a Go language change) are
  paid in small, frequent increments instead of one large stall.
- CI cost rises (every bump rebuilds/tests every affected module). Accepted.
</content>
