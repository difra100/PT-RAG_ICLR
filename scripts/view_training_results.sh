#!/bin/bash
#
# Script per visualizzare rapidamente le metriche di training
# Uso: ./view_training_results.sh [output_dir]
#

set -e

# Default directory
OUTPUT_DIR="${1:-$HOME/PT-RAG/debug_test/debug_test}"

echo "=================================="
echo "STATE TRAINING RESULTS VIEWER"
echo "=================================="
echo ""
echo "📁 Directory: $OUTPUT_DIR"
echo ""

# Verifica che la directory esista
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Directory non trovata: $OUTPUT_DIR"
    exit 1
fi

# Trova la directory version_*
VERSION_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "version_*" | head -1)

if [ -z "$VERSION_DIR" ]; then
    echo "❌ Nessuna directory version_* trovata in $OUTPUT_DIR"
    exit 1
fi

echo "📊 Version: $(basename $VERSION_DIR)"
echo ""

# Metriche CSV
METRICS_CSV="$VERSION_DIR/metrics.csv"
if [ -f "$METRICS_CSV" ]; then
    echo "📈 METRICHE (CSV):"
    echo "-----------------------------------"
    column -t -s, "$METRICS_CSV"
    echo ""
else
    echo "⚠️  metrics.csv non trovato"
    echo ""
fi

# Hyperparameters
HPARAMS="$VERSION_DIR/hparams.yaml"
if [ -f "$HPARAMS" ]; then
    echo "⚙️  HYPERPARAMETERS (primi 40):"
    echo "-----------------------------------"
    head -40 "$HPARAMS"
    echo "..."
    echo ""
else
    echo "⚠️  hparams.yaml non trovato"
    echo ""
fi

# Checkpoints
CKPT_DIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    echo "💾 CHECKPOINTS:"
    echo "-----------------------------------"
    ls -lh "$CKPT_DIR" | grep -v "^total" | awk '{print $9, "(" $5 ")"}'
    echo ""
    
    # Conta i checkpoint
    NUM_CKPTS=$(ls -1 "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l)
    echo "  Total checkpoints: $NUM_CKPTS"
    echo ""
else
    echo "⚠️  checkpoints/ directory non trovata"
    echo ""
fi

# Config
CONFIG="$OUTPUT_DIR/config.yaml"
if [ -f "$CONFIG" ]; then
    echo "📄 CONFIG disponibile:"
    echo "  $CONFIG"
    echo ""
fi

# Summary
echo "=================================="
echo "SUMMARY"
echo "=================================="

# Leggi ultima loss dal CSV (se esiste)
if [ -f "$METRICS_CSV" ]; then
    LAST_VAL_LOSS=$(tail -1 "$METRICS_CSV" | cut -d',' -f5)
    LAST_STEP=$(tail -1 "$METRICS_CSV" | cut -d',' -f3)
    
    echo "  Last validation step: $LAST_STEP"
    echo "  Last val_loss: $LAST_VAL_LOSS"
    echo ""
fi

echo "=================================="
echo ""
echo "💡 Per visualizzare in dettaglio:"
echo "   python ~/PT-RAG/scripts/view_metrics.py $METRICS_CSV"
echo ""
echo "💡 Per aprire in VS Code:"
echo "   code $OUTPUT_DIR"
echo ""
