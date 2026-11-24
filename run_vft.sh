#!/usr/bin/env bash
# Stage 2: verdict fine-tuning (VFT) -- full-parameter training of the candidate scoring head.
set -euo pipefail
export STAGE=full
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/supervised_base.sh" "$@"
