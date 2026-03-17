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

ensure_se3_editable_install() {
	local repo_dir="$1"
	local se3_dir="${SE3TRANSFORMER_SRC:-$repo_dir/env/SE3Transformer}"

	if [ ! -f "$se3_dir/setup.py" ] || [ ! -d "$se3_dir/se3_transformer" ]; then
		return 0
	fi

	if ! SE3_DIR="$se3_dir" python - <<'PY'
import os
import sys

se3_dir = os.path.realpath(os.environ["SE3_DIR"])

try:
	import se3_transformer
except Exception:
	sys.exit(1)

module_path = os.path.realpath(getattr(se3_transformer, "__file__", ""))
if not module_path.startswith(se3_dir + os.sep):
	sys.exit(1)
PY
	then
		echo "Installing SE3Transformer in editable mode from $se3_dir"
		pip install -e "$se3_dir"
	fi
}

if repo_dir="$(find_repo_dir)"; then
	export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
	ensure_editable_install "$repo_dir"
	ensure_se3_editable_install "$repo_dir"
fi

if [ "$#" -gt 0 ]; then
	exec "$@"
fi

exec bash