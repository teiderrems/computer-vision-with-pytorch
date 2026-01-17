#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {serve|list|execute}"
  echo "  serve   : lance Jupyter Lab (interface interactive)"
  echo "  list    : liste les notebooks (*.ipynb) trouvés (récursif)"
  echo "  execute : exécute tous les notebooks trouvés (inplace) via nbconvert"
  echo "            NBEXEC_TIMEOUT env var peut ajuster le timeout (en secondes, -1 pour illimité)"
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

MODE="$1"

# Exclure ces chemins communs
EXCLUDE_PATTERNS=("*/.ipynb_checkpoints/*" "./.venv/*" "./venv/*" "./env/*")

find_notebooks() {
  local find_cmd=(find . -type f -name "*.ipynb")
  for p in "${EXCLUDE_PATTERNS[@]}"; do
    find_cmd+=( -not -path "$p" )
  done
  find_cmd+=( -print0 )
  "${find_cmd[@]}"
}

case "$MODE" in
  serve)
    echo "Lancement de Jupyter Lab..."
    jupyter lab
    ;;
  list)
    echo "Recherche des notebooks..."
    while IFS= read -r -d '' nb; do
      printf "%s\n" "$nb"
    done < <(find_notebooks | sort -z)
    ;;
  execute)
    echo "Exécution des notebooks (cela peut prendre du temps)..."
    TIMEOUT="${NBEXEC_TIMEOUT:--1}"
    while IFS= read -r -d '' nb; do
      printf "-> Exécution : %s\n" "$nb"
      jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout="$TIMEOUT" "$nb"
    done < <(find_notebooks | sort -z)
    echo "Exécution terminée."
    ;;
  *)
    usage
    ;;
esac
