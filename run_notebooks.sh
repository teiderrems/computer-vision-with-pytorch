#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {serve|execute}"
  echo "  serve   : lance Jupyter Lab (interface interactive)"
  echo "  execute : exécute tous les notebooks (inplace) dans td1..td4 via nbconvert" 
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

MODE="$1"

case "$MODE" in
  serve)
    echo "Lancement de Jupyter Lab..."
    jupyter lab
    ;;
  execute)
    echo "Exécution des notebooks (cela peut prendre du temps)..."
    for nb in td1/*.ipynb td2/*.ipynb td3/*.ipynb td4/*.ipynb; do
      [ -f "$nb" ] || continue
      echo "-> Exécution : $nb"
      jupyter nbconvert --to notebook --execute --inplace "$nb"
    done
    echo "Exécution terminée."
    ;;
  *)
    usage
    ;;
esac
