#!/bin/bash
# Prepare this sandbox's /work volume to seed other sandboxes through a file-system snapshot (DiLoCo coordinator and
# learners restored from it), so nothing large crosses the controller's uplink.
#
#   seed_volume.sh PARENT_CHECKPOINT DEST_DIR
#
# 1. Copies the parent full-state checkpoint (and its policy sidecar) to DEST_DIR on /work, verified by sha256.
# 2. Stores every object of this sandbox's bundle (code archive and the fetched input files) in /work/.coreweave/cas,
#    keyed by sha256, so cwctl.push_inputs links them instead of pushing them again.
# 3. Removes this sandbox's bundle state (inputs-ready marker, bundle.json, receipts, outputs) so a restored sandbox's
#    bootstrap waits for its own inputs.
# Prints the parent sha256 and the cas inventory size.
set -euo pipefail
PARENT=$1; DEST=$2
CW=/work/.coreweave
mkdir -p "$DEST" "$CW/cas"
cp "$PARENT" "$DEST/checkpoint.pt.tmp" && mv "$DEST/checkpoint.pt.tmp" "$DEST/checkpoint.pt"
[ -f "$(dirname "$PARENT")/checkpoint-policy.json" ] && cp "$(dirname "$PARENT")/checkpoint-policy.json" "$DEST/"
a=$(sha256sum "$PARENT" | cut -d' ' -f1); b=$(sha256sum "$DEST/checkpoint.pt" | cut -d' ' -f1)
[ "$a" = "$b" ] || { echo "parent copy mismatch" >&2; exit 1; }
python3 - "$CW" "${NATLANG_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" <<'PY'
import hashlib, json, os, shutil, sys
cw, repo = sys.argv[1], sys.argv[2]
manifest = json.load(open(f'{cw}/bundle.json'))
stored = 0
for item in manifest['files']:
    path = item['path'] if os.path.isabs(item['path']) else os.path.join(repo, item['path'])
    target = f'{cw}/cas/{item["sha256"]}'
    if os.path.exists(target) or not os.path.isfile(path):
        continue
    with open(path, 'rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != item['sha256']:
            continue  # changed since the fetch: not the bundle object any more
    shutil.copyfile(path, target + '.tmp'); os.replace(target + '.tmp', target); stored += 1
code = manifest.get('code')
if code and os.path.exists(f'{cw}/code.tar.gz') and not os.path.exists(f'{cw}/cas/{code["sha256"]}'):
    os.link(f'{cw}/code.tar.gz', f'{cw}/cas/{code["sha256"]}'); stored += 1
print(json.dumps({'cas_stored': stored, 'cas_objects': len(os.listdir(f'{cw}/cas'))}))
PY
rm -rf "$CW/inputs-ready" "$CW/bundle.json" "$CW/receipts" "$CW/outputs" "$CW/outputs-ready" "$CW/outputs-collected" \
       "$CW/exit-status" "$CW/inbox" "$CW/secrets" "$CW/code.tar.gz"
sync
echo "{\"parent\": \"$DEST/checkpoint.pt\", \"sha256\": \"$b\", \"bytes\": $(stat -c %s "$DEST/checkpoint.pt")}"
