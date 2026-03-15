#!/usr/bin/env bash
set -euo pipefail

BRANCH_NAME=${1:-sprint05-docs-and-session-eval}

pytest -q

git checkout -b "$BRANCH_NAME"
git add .
git commit -m "Sprint 05: spot packs, sessions, traces persistence, full docs"
echo "Review commit and push to your remote when ready."
