#!/bin/bash
# watch_log.sh — Sigue el log más reciente de un job SLURM en tiempo real
#
# Uso:
#   bash hercules/watch_log.sh              # último log de logs/
#   bash hercules/watch_log.sh noise        # último log que contenga "noise"
#   bash hercules/watch_log.sh quantics     # último log que contenga "quantics"

PATTERN="${1:-}"          # filtro opcional (noise, quantics, seed...)
LOG_DIR="logs"
INTERVAL=2                # segundos entre comprobaciones si el fichero no existe aún

# ── Buscar el log más reciente ──────────────────────────────────────────────
find_latest_log() {
    if [ -n "$PATTERN" ]; then
        find "$LOG_DIR" -name "*${PATTERN}*.out" -printf "%T@ %p\n" 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}'
    else
        find "$LOG_DIR" -name "*.out" -printf "%T@ %p\n" 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}'
    fi
}

# ── Esperar a que aparezca el log ───────────────────────────────────────────
echo "Buscando logs en $LOG_DIR/ (patrón: '${PATTERN:-cualquiera}')..."

LOG=""
while [ -z "$LOG" ]; do
    LOG=$(find_latest_log)
    if [ -z "$LOG" ]; then
        printf "\r  Esperando log...  $(date +%H:%M:%S)"
        sleep $INTERVAL
    fi
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Log: $LOG"
echo "  $(date)"
echo "══════════════════════════════════════════════════════"

# ── Estado del job en SLURM ─────────────────────────────────────────────────
if command -v squeue &>/dev/null; then
    echo ""
    echo "Jobs activos:"
    squeue -u "$USER" --format="  %-10i %-12j %-8T %-10M %-6C %R" 2>/dev/null \
        || echo "  (squeue no disponible en este nodo)"
    echo ""
fi

# ── Seguir el log en tiempo real ────────────────────────────────────────────
# Muestra las últimas 50 líneas y sigue el fichero
tail -n 50 -f "$LOG"
