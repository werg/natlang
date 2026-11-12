#!/bin/sh
# Install the natlang command line: curl -fsSL https://raw.githubusercontent.com/werg/natlang/main/scripts/install.sh | sh
#
# Needs Node.js >= 22.13 with npm. Installs @natlang/cli and @natlang/node from the newest GitHub release
# (tag v*) with `npm install --global`, into npm's global prefix when it is writable and ~/.local otherwise.
# Settings (environment):
#   NATLANG_VERSION  a release tag (v0.1.0) instead of the newest one
#   NATLANG_FROM     a directory or URL holding natlang-node.tgz and natlang-cli.tgz (local builds, mirrors)
#   NATLANG_PACKAGE  an npm spec to install instead of release tarballs, e.g. @natlang/cli
#   NATLANG_PREFIX   the npm prefix to install into (the command lands in PREFIX/bin)
set -eu

fail() { printf 'natlang install: %s\n' "$1" >&2; exit 1; }

command -v node >/dev/null 2>&1 || fail 'Node.js 22.13 or newer is required: https://nodejs.org/en/download'
command -v npm >/dev/null 2>&1 || fail 'npm is required (it ships with Node.js): https://nodejs.org/en/download'
node -e 'const [a, b] = process.versions.node.split(".").map(Number); process.exit(a > 22 || (a === 22 && b >= 13) ? 0 : 1)' ||
  fail "Node.js 22.13 or newer is required (found $(node --version)): https://nodejs.org/en/download"

if [ -n "${NATLANG_PACKAGE:-}" ]; then
  packages=$NATLANG_PACKAGE
else
  from=${NATLANG_FROM:-}
  if [ -z "$from" ]; then
    # The release that carries the CLI: the named tag, else the newest v* release (other releases hold models).
    from=$(NATLANG_VERSION=${NATLANG_VERSION:-} node -e '
      const tag = process.env.NATLANG_VERSION;
      const base = "https://github.com/werg/natlang/releases/download/";
      if (tag) { console.log(base + tag); process.exit(0); }
      fetch("https://api.github.com/repos/werg/natlang/releases?per_page=100", { headers: { accept: "application/vnd.github+json" } })
        .then(response => { if (!response.ok) throw new Error(`GitHub answered ${response.status}`); return response.json(); })
        .then(releases => {
          const release = releases.find(item => /^v\d/.test(item.tag_name) && !item.draft && !item.prerelease &&
            item.assets.some(asset => asset.name === "natlang-cli.tgz"));
          if (!release) throw new Error("no natlang release with natlang-cli.tgz yet; install from source instead: " +
            "git clone https://github.com/werg/natlang && cd natlang && scripts/setup_dev.sh --node-only");
          console.log(base + release.tag_name);
        })
        .catch(error => { console.error(`natlang install: ${error.message}`); process.exit(1); });') || exit 1
  fi
  case $from in
    http://*|https://*) ;;
    *) from=$(cd "$from" && pwd) || fail "NATLANG_FROM is not a directory: $NATLANG_FROM" ;;
  esac
  packages="$from/natlang-node.tgz $from/natlang-cli.tgz"
fi

prefix=${NATLANG_PREFIX:-}
if [ -z "$prefix" ]; then
  prefix=$(npm prefix --global)
  # A system Node's prefix (/usr, /usr/local) is not writable without sudo: install for this user instead.
  if [ ! -w "$prefix/lib" ] && [ ! -w "$prefix" ]; then prefix=$HOME/.local; fi
fi

printf 'Installing natlang into %s ...\n' "$prefix"
# shellcheck disable=SC2086 # packages is a word list
npm install --global --prefix "$prefix" --no-fund --no-audit --loglevel=error $packages
"$prefix/bin/natlang" --version >/dev/null || fail "the installed command did not start: $prefix/bin/natlang"

case ":$PATH:" in
  *":$prefix/bin:"*) ;;
  *) printf '\nAdd natlang to your PATH (and to your shell profile):\n  export PATH="%s/bin:$PATH"\n' "$prefix" ;;
esac
printf '\nnatlang is installed. Next:\n  natlang init my-program && cd my-program\n  natlang setup    # the local model runtime, once (about 2 GB); or configure a hosted model: natlang dashboard\n  natlang run\n'
