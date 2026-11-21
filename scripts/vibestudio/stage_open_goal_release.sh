#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 || ! $3 =~ ^[a-f0-9]{64}$ ]]; then
  printf 'Usage: %s PINS.tsv RELEASE_DIRECTORY CATALOG_SHA256\n' "$0" >&2
  exit 2
fi
pins=$(realpath "$1")
release=$(realpath -m "$2")
expected_catalog_sha=$3
stage_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
host_source=${VIBESTUDIO_SOURCE_CHECKOUT:-"$HOME/vibestudio"}
template_sources=${VIBESTUDIO_TEMPLATE_SOURCE_ROOT:-"$HOME/vibestudio-templates"}
natlang_source=${NATLANG_SOURCE_CHECKOUT:-"$HOME/natlang"}

declare -A commits
while IFS=$'\t' read -r name sha extra; do
  case "$name" in
    natlang | host | templates/base | templates/system-testing | templates/system | templates/examples | \
      templates/google-workspace | templates/news | templates/personal | templates/spectrolite) ;;
    *) printf 'Unknown repository in pins: %s\n' "$name" >&2; exit 2 ;;
  esac
  if [[ ! $sha =~ ^[a-f0-9]{40}$ || -n $extra || -v commits[$name] ]]; then
    printf 'Expected exactly one pinned commit for %s\n' "$name" >&2
    exit 2
  fi
  commits[$name]=$sha
done < "$pins"
if [[ ${#commits[@]} != 10 ]]; then
  printf 'Pins must cover Natlang, Vibestudio and all eight templates\n' >&2
  exit 2
fi

mkdir -p "$release/templates"
for name in natlang host templates/{base,system-testing,system,examples,google-workspace,news,personal,spectrolite}; do
  case "$name" in
    natlang) source=$natlang_source ;;
    host) source=$host_source ;;
    *) source="$template_sources/${name#templates/}" ;;
  esac
  sha=${commits[$name]}
  if [[ ! -e "$release/$name" ]]; then
    if ! git -C "$source" cat-file -e "$sha^{commit}" 2>/dev/null; then
      git -C "$source" fetch origin main
    fi
    git -C "$source" worktree add --detach "$release/$name" "$sha"
  fi
  test "$(git -C "$release/$name" rev-parse HEAD)" = "$sha"
  test -z "$(git -C "$release/$name" status --porcelain)"
done

cd "$release/host"
export VIBESTUDIO_TEMPLATE_CHECKOUTS="$release/templates"
export NATLANG_REPO="$release/natlang"
export npm_config_workspace_concurrency=1
export NODE_OPTIONS=${NODE_OPTIONS:-"--max-old-space-size=850 --max-semi-space-size=8"}
pnpm install --frozen-lockfile --ignore-scripts
pnpm check:native-host-dependencies --repair
if [[ -n ${VIBESTUDIO_INFRASTRUCTURE_DONOR:-} ]]; then
  node "$stage_scripts/reuse_host_infrastructure.mjs" "$VIBESTUDIO_INFRASTRUCTURE_DONOR" "$PWD"
fi
pnpm build:infrastructure
node --import tsx --input-type=module -e 'await import("./packages/rpc/src/wireJson.ts")'
pnpm natlang:vendor --check
node --import tsx scripts/check-open-goal-runtime.mjs
node --import tsx scripts/check-open-goal-runtime.mjs system+system-testing
catalog=$(mktemp "$release/catalog.XXXXXX")
trap 'if [[ -f "$catalog" ]]; then unlink "$catalog"; fi' EXIT
pnpm --silent natlang:teacher --list > "$catalog"
actual_catalog_sha=$(sha256sum "$catalog")
test "${actual_catalog_sha%% *}" = "$expected_catalog_sha"
mv "$catalog" "$release/catalog.jsonl"
test -z "$(git status --porcelain)"
(
  cd "$release/natlang"
  npm ci --ignore-scripts --prefer-offline --no-audit
  npm --prefix ts-host run --silent build:node
  node --input-type=module -e '
    const { createPiModelBackend } = await import("./ts-host/dist/model/pi-provider.js");
    const { modelOracleJudge } = await import("./ts-host/dist/teacher/model-judge.js");
    if (typeof createPiModelBackend !== "function" || typeof modelOracleJudge !== "function") process.exit(1);
  '
  test -z "$(git status --porcelain)"
)
touch "$release/staged"
printf 'Verified %s catalog entries in %s\n' "$(wc -l < "$release/catalog.jsonl")" "$release"
