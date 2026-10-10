#!/bin/sh
# Build libphotonstyx.so for the Raze board (aarch64 Linux, glibc 2.41) into
# photon-styx/target/aarch64-unknown-linux-gnu/release/libphotonstyx.so.
#   tools/build.sh            release build for the board
#   tools/build.sh host       host build and unit tests (x86_64)
#
# EIDOS_PATH=<dir>: build against a local Eidos checkout (a directory at the pinned commit, e.g. a
# worktree) instead of the git dependency. Every eidos-* package in <dir>/crates is redirected with
# a cargo --config patch, so the build never reads another checkout or fetches Eidos from GitHub.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"
target_dir=${CARGO_TARGET_DIR:-$here/target}

# Cargo --config arguments for the Eidos patch, as one string for eval.
patch_args=""
if [ -n "${EIDOS_PATH:-}" ]; then
  for manifest in "$EIDOS_PATH"/crates/*/Cargo.toml; do
    name=$(sed -n 's/^name = "\(.*\)"$/\1/p' "$manifest" | head -n 1)
    dir=$(dirname "$manifest")
    patch_args="$patch_args --config 'patch.\"https://github.com/Prometheus-Dynamics/Eidos.git\".$name.path=\"$dir\"'"
  done
  echo "patching $(echo "$patch_args" | grep -o -- "--config" | wc -l) eidos crates from $EIDOS_PATH"
fi

cargo_run() {
  eval "CARGO_TARGET_DIR=\"$target_dir\" cargo $*$patch_args"
}

if [ "${1:-}" = "host" ]; then
  cargo_run test --lib
  exit 0
fi
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$here/tools/zig-cc-aarch64.sh"
export CC_aarch64_unknown_linux_gnu="$here/tools/zig-cc-aarch64.sh"
export CXX_aarch64_unknown_linux_gnu="$here/tools/zig-c++-aarch64.sh"
export AR_aarch64_unknown_linux_gnu=ar
cargo_run build --release --target aarch64-unknown-linux-gnu --lib
echo "built $target_dir/aarch64-unknown-linux-gnu/release/libphotonstyx.so"
