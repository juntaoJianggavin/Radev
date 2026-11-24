#!/usr/bin/env bash
# Stage 1: projector--evidence alignment (LLM + vision tower frozen; bridging modules only).
set -euo pipefail
export STAGE=align
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/supervised_base.sh" "$@"
