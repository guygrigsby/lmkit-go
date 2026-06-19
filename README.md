# lmkit-go

> The Go sibling to [lmkit](https://github.com/guygrigsby/lmkit). Aeryx/`guygrigsby` (research), MIT.

From-scratch LLM training in pure Go, on an XLA backend. An OSS give-back: the
ergonomic Go training toolkit that the Go ecosystem is missing, built *on top of*
a real compiler (XLA via GoMLX) rather than reinventing kernels.

**Status:** design only. Start at [`DESIGN.md`](./DESIGN.md).

**Not** a replacement for the Python/PyTorch training stack ([lmkit](https://github.com/guygrigsby/lmkit));
that stays. This is a separate, for-fun, community-facing project.

**North-star artifact:** reproduce a published tiny-Llama baseline
(`Aeryx-ai/lm-100m-en-overtrained`, val loss 1.7337) in pure Go-on-XLA. Tiny
model, known target, dogfooded against real numbers.
