# 8. Public API evolves additively

Status: accepted (2026-06-19)

## Context

Per-package public modules (ADR-0005) are meant for reuse, versioned with
`module/vX.Y.Z` tags. The moment others depend on a tagged version, every exported
identifier is a contract. Meanwhile interfaces like `backend` are deliberately
"grown, not designed" — minimal now, expanded as `model`/`train` need ops.

Growing is safe; *removing* is not. Dropping or changing the signature of an
exported symbol breaks every downstream consumer and forces a major-version bump.
The "expand later" plan only works if expansion is additive.

## Decision

Public APIs evolve additively within a major version. Concretely:

- **Add, don't remove or re-sign.** New methods/types/fields are fine (minor bump).
  Removing an exported symbol, changing a signature, or changing observable
  behavior is breaking and requires a major-version bump (SemVer).
- **Under-expose first.** When in doubt, keep a new type/method unexported until a
  consumer actually needs it. It is cheap to export later, expensive to unexport.
  This directly governs the minimal `backend` interface: prefer too small over a
  shape we must break.
- **Grow interfaces by addition or by new interfaces**, not by editing a published
  one out from under implementers. A wide published interface is hard to extend
  without breaking implementers, so keep published interfaces small and focused.
- **Deprecate before remove.** Mark with a `// Deprecated:` doc comment and keep the
  symbol through the current major version; remove only at the next major.

## Consequences

- The "grow, not design" approach (Milestone 1) is safe *because* growth is
  additive — the constraint and the approach are consistent.
- Designing minimal now is a feature: less exported surface = less we are bound to
  keep. Bias toward unexported until proven needed.
- A genuine breaking change is allowed — it costs a major-version bump and a
  migration note, paid deliberately, not slipped into a minor release.
- Applies to every public module (backend, tokenizer, safetensors, gguf, hub,
  data, model, train, io) — not just `backend`.
</content>
