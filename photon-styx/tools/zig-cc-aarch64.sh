#!/bin/sh
# C compiler and linker for aarch64-unknown-linux-gnu, pinned to glibc 2.41: the Raze image's
# Buildroot glibc. Any symbol newer than that would fail to load on the board. cmake (through
# turbojpeg-sys) passes its own --target, which zig would combine with ours: drop it. rustc's
# A53 erratum flag and its gcc-ld selection are GNU-linker options that zig's linker does not take.
args=""
for a in "$@"; do
  case "$a" in
    --target=*|-Wl,--fix-cortex-a53-843419|-fuse-ld=*|-B*) ;;
    *) args="$args
$a" ;;
  esac
done
set -- $(printf '%s' "$args" | sed '/^$/d' | tr '\n' '\001' | sed 's/\x01/ /g')
exec zig cc -target aarch64-linux-gnu.2.41 "$@"
