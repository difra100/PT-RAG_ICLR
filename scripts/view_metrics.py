#!/usr/bin/env python3
"""
Script per visualizzare le metriche di training salvate da PyTorch Lightning
"""
import pandas as pd
import sys
from pathlib import Path

def view_metrics(metrics_path):
    """Visualizza le metriche da un file CSV"""
    
    # Carica il CSV
    df = pd.read_csv(metrics_path)
    
    print("=" * 80)
    print(f"METRICHE DI TRAINING: {metrics_path}")
    print("=" * 80)
    print()
    
    # Info generali
    print(f"📊 Numero totale di validation steps: {len(df)}")
    print(f"📈 Colonne disponibili: {', '.join(df.columns)}")
    print()
    
    # Mostra tutte le righe
    print("📋 METRICHE COMPLETE:")
    print("-" * 80)
    print(df.to_string(index=False))
    print()
    
    # Statistiche
    if 'val_loss' in df.columns:
        print("📉 STATISTICHE VAL_LOSS:")
        print("-" * 80)
        print(f"  Initial:  {df['val_loss'].iloc[0]:.4f}")
        print(f"  Final:    {df['val_loss'].iloc[-1]:.4f}")
        print(f"  Min:      {df['val_loss'].min():.4f}")
        print(f"  Max:      {df['val_loss'].max():.4f}")
        print(f"  Mean:     {df['val_loss'].mean():.4f}")
        print(f"  Std:      {df['val_loss'].std():.4f}")
        
        # Trend
        improvement = df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]
        if improvement > 0:
            print(f"  ✅ Improvement: {improvement:.4f} (loss decreased)")
        else:
            print(f"  ⚠️  Change: {improvement:.4f} (loss increased)")
        print()
    
    if 'val/decoder_loss' in df.columns:
        print("📉 STATISTICHE DECODER_LOSS:")
        print("-" * 80)
        print(f"  Initial:  {df['val/decoder_loss'].iloc[0]:.4f}")
        print(f"  Final:    {df['val/decoder_loss'].iloc[-1]:.4f}")
        print(f"  Min:      {df['val/decoder_loss'].min():.4f}")
        print(f"  Max:      {df['val/decoder_loss'].max():.4f}")
        print(f"  Mean:     {df['val/decoder_loss'].mean():.4f}")
        print()
    
    # FLOPs se disponibili
    if 'cumulative_flops_val_sync' in df.columns:
        total_flops = df['cumulative_flops_val_sync'].iloc[-1]
        print("💻 COMPUTE:")
        print("-" * 80)
        print(f"  Total FLOPs: {total_flops:.2e}")
        print(f"  Total GFLOPs: {total_flops / 1e9:.2f}")
        print(f"  Total TFLOPs: {total_flops / 1e12:.4f}")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        metrics_path = sys.argv[1]
    else:
        # Default path
        metrics_path = Path.home() / "PT-RAG/debug_test/debug_test/version_0/metrics.csv"
    
    if not Path(metrics_path).exists():
        print(f"❌ File non trovato: {metrics_path}")
        print()
        print("Uso: python view_metrics.py [path/to/metrics.csv]")
        print()
        print("Esempio:")
        print(f"  python view_metrics.py ~/PT-RAG/debug_test/debug_test/version_0/metrics.csv")
        sys.exit(1)
    
    view_metrics(metrics_path)
