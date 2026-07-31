#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --python PYTHON --inputs-json ABS --expected-inputs-sha256 SHA256 --output-json ABS" >&2
  exit 2
}

python_bin=""
inputs_json=""
inputs_sha=""
output_json=""
while (($#)); do
  case "$1" in
    --python) python_bin="${2-}"; shift 2 ;;
    --inputs-json) inputs_json="${2-}"; shift 2 ;;
    --expected-inputs-sha256) inputs_sha="${2-}"; shift 2 ;;
    --output-json) output_json="${2-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "$python_bin" && -n "$inputs_json" && -n "$inputs_sha" && -n "$output_json" ]] || usage
[[ "$inputs_json" == /* && "$output_json" == /* ]] || usage

exec "$python_bin" -m scripts.show_base.base_final_authority \
  --inputs-json "$inputs_json" \
  --expected-inputs-sha256 "$inputs_sha" \
  --output-json "$output_json"
