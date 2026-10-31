#!/bin/sh
# usage: verify_source.sh BUILD_DIR SOURCE [CONCURRENCY] -- qualify BUILD_DIR/SOURCE/raw-cases.jsonl into BUILD_DIR/SOURCE/verified/
set -eu
B=$1; S=$2; C=${3:-8}
export PATH=/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin:$PATH
export NATLANG_RL_TASK_ASSETS=$B/../assets NATLANG_HASH_CACHE=$B/$S/hash-cache.jsonl NODE_OPTIONS=--max-old-space-size=2000
cd /home/werg/natlang
exec node scripts/rl_tasks/verify.mjs /home/werg/natlang/ts-host $B/$S/raw-cases.jsonl $B/$S/verified $C
