#!/usr/bin/env bash
# Install the qemu user-mode emulators so `qemu-aarch64-static` and
# `qemu-x86_64-static` resolve on every Ubuntu runner image.
#
# Ubuntu 26.04 dropped the `qemu-user-static` package (it is now a virtual
# package with no installation candidate, and `apt-get install` exits 100).
# There, `qemu-user` carries the emulators under their plain names, so this
# script links the `-static` names the parity scripts call. On images that
# still ship `qemu-user-static` it installs that, as before.
#
# Usage: scripts/install-qemu-user.sh [extra apt packages...]
# Set QEMU_INSTALL_FORCE_PLAIN=1 to take the `qemu-user` path on any image.
set -euo pipefail

sudo apt-get update
if [ "${QEMU_INSTALL_FORCE_PLAIN:-0}" != 1 ] \
   && apt-cache show qemu-user-static 2>/dev/null | grep -q '^Filename:'; then
  sudo apt-get install -y "$@" qemu-user-static
else
  sudo apt-get install -y "$@" qemu-user
fi

for arch in aarch64 x86_64; do
  if [ "${QEMU_INSTALL_FORCE_PLAIN:-0}" = 1 ] || ! command -v "qemu-$arch-static" >/dev/null; then
    real=$(command -v "qemu-$arch") || { echo "qemu-$arch is not installed" >&2; exit 1; }
    sudo ln -sf "$real" "/usr/local/bin/qemu-$arch-static"
  fi
  echo "qemu-$arch-static -> $(readlink -f "$(command -v "qemu-$arch-static")")"
  "qemu-$arch-static" --version | head -1
done
