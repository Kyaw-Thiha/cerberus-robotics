#!/usr/bin/env bash
# The 60 s EGO -> CHASE -> OVERVIEW cut and the 10 s teaser GIF, from clips that passed the gate.
# Usage: make_hero_cut.sh <clip dir> [ego.mp4 chase.mp4 overview.mp4]
set -euo pipefail
D="${1:?clip dir}"
EGO="${2:-walk_ego.mp4}"; CHASE="${3:-walk_chase.mp4}"; OVERVIEW="${4:-walk_overview.mp4}"
cd "$D"
printf "file '%s'\nfile '%s'\nfile '%s'\n" "$EGO" "$CHASE" "$OVERVIEW" > hero_concat.txt
ffmpeg -y -f concat -safe 0 -i hero_concat.txt -c copy hero_walk_60s.mp4
ffmpeg -y -ss 2 -t 10 -i "$CHASE" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" teaser_walk_10s.gif
ls -l hero_walk_60s.mp4 teaser_walk_10s.gif
