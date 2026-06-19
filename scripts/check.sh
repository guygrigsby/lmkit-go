#!/usr/bin/env bash
# Build and test every module on the pure-Go SimpleGo backend (no XLA libs).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for mod in backend app; do
  echo "== $mod =="
  ( cd "$mod" && go build -tags noxla ./... && GOMLX_BACKEND=go go test -tags noxla ./... )
done
echo "== boundary =="
go test -tags noxla ./app/ -run TestBackendBoundary
echo "OK"
