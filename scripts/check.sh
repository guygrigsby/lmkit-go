#!/usr/bin/env bash
# Build and test every module on the pure-Go SimpleGo backend (no XLA libs).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for mod in backend model app; do
  echo "== $mod =="
  ( cd "$mod" && go build -tags noxla ./... && GOMLX_BACKEND=go go test -tags noxla ./... )
done
echo "== boundary =="
# -count=1: the boundary test shells out to `git grep`, which `go test` does not
# track as a cache input, so a cached pass would hide a leak introduced in another
# package. Force a real run every time.
go test -count=1 -tags noxla ./app/ -run TestBackendBoundary
echo "OK"
