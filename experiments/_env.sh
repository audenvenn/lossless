# Sourced by the run_*.sh scripts. Loads machine-local model/engine paths (untracked paths.local.env, see paths.py)
# and forbids implicit Hugging Face downloads, so a missing local path fails loudly instead of fetching gigabytes.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_here/paths.local.env" ]; then set -a; . "$_here/paths.local.env"; set +a; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
