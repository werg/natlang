#!/bin/sh
git log --merges --pretty='- %s' "$1"..HEAD
