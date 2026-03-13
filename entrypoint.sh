#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate rfdiffusion

find_repo_dir() {
	local candidate

	for candidate in "$RFDIFFUSION_SRC" "$PWD" /workspace/RFdiffusion; do
		if [ -n "$candidate" ] && [ -f "$candidate/setup.py" ] && [ -d "$candidate/rfdiffusion" ]; then
			printf '%s\n' "$candidate"
			return 0
		fi
	done

	return 1
}

ensure_editable_install() {
	local repo_dir="$1"

	if ! REPO_DIR="$repo_dir" python - <<'PY'
import os
import sys

repo_dir = os.path.realpath(os.environ["REPO_DIR"])

try:
	import rfdiffusion
except Exception:
	sys.exit(1)

module_path = os.path.realpath(getattr(rfdiffusion, "__file__", ""))
if not module_path.startswith(repo_dir + os.sep):
	sys.exit(1)
PY
	then
		echo "Installing RFdiffusion in editable mode from $repo_dir"
		pip install -e "$repo_dir"
	fi
}

if repo_dir="$(find_repo_dir)"; then
	export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
	ensure_editable_install "$repo_dir"
fi

if [ "$#" -gt 0 ]; then
	exec "$@"
fi

exec bash