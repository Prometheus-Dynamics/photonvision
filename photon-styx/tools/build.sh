#!/bin/sh
# Build libphotonstyx.so for the Raze board (aarch64 Linux, glibc 2.41) into
# photon-styx/target/aarch64-unknown-linux-gnu/release/libphotonstyx.so.
#   tools/build.sh            release build for the board
#   tools/build.sh host       host build and unit tests (x86_64)
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"
target_dir=${CARGO_TARGET_DIR:-$here/target}
if [ "${1:-}" = "host" ]; then
  CARGO_TARGET_DIR="$target_dir" cargo test --lib
  exit 0
fi
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$here/tools/zig-cc-aarch64.sh"
export CC_aarch64_unknown_linux_gnu="$here/tools/zig-cc-aarch64.sh"
export CXX_aarch64_unknown_linux_gnu="$here/tools/zig-c++-aarch64.sh"
export AR_aarch64_unknown_linux_gnu=ar
CARGO_TARGET_DIR="$target_dir" cargo build --release --target aarch64-unknown-linux-gnu --lib
echo "built $target_dir/aarch64-unknown-linux-gnu/release/libphotonstyx.so"
