#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$DIR"

NSL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
UNSW="https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification"

fetch () {
  if [ -s "$2" ]; then
    echo "have    $(basename "$2")"
  else
    echo "fetch   $(basename "$2")"
    curl -sSL --fail -o "$2" "$1"
  fi
}

fetch "$NSL/KDDTrain%2B.txt"              "$DIR/KDDTrain+.txt"
fetch "$NSL/KDDTest%2B.txt"               "$DIR/KDDTest+.txt"
fetch "$UNSW/UNSW_NB15_training-set.csv"  "$DIR/UNSW_NB15_training-set.csv"
fetch "$UNSW/UNSW_NB15_testing-set.csv"   "$DIR/UNSW_NB15_testing-set.csv"

echo
echo "--- expected sizes ---"
echo "KDDTrain+.txt              125973 rows"
echo "KDDTest+.txt                22544 rows"
echo "UNSW_NB15_training-set.csv 175341 rows"
echo "UNSW_NB15_testing-set.csv   82332 rows"
echo
wc -l "$DIR"/*.txt "$DIR"/*.csv
