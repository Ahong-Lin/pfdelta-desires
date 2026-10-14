#!/usr/bin/env bash
set -euo pipefail
# Optional user-space installation; no sudo, no shell profile changes.
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)
    url=https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
    expected=07d20c4c2518833e2265ca0acee15b355463361aa4efdab858dad826cf94325c
    ;;
  *) echo 'This optional installer supports Linux x86_64. Install Julia 1.9.4 for your platform and set JULIA_BIN.' >&2; exit 1 ;;
esac
destination="$bundle_dir/.runtime"
if [[ -x "$destination/julia-1.9.4/bin/julia" ]]; then
  "$destination/julia-1.9.4/bin/julia" --version
  exit 0
fi
mkdir -p "$destination"
archive="$destination/julia-1.9.4.tar.gz"
curl --fail --location --retry 3 --output "$archive.part" "$url"
actual=$(sha256sum "$archive.part" | cut -d' ' -f1)
[[ "$actual" == "$expected" ]] || { echo 'Julia archive SHA256 mismatch' >&2; exit 1; }
mv "$archive.part" "$archive"
tar -xzf "$archive" -C "$destination"
"$destination/julia-1.9.4/bin/julia" --version
