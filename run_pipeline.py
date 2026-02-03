#!/usr/bin/env python3
"""
run_pipeline.py
Entry point principale per l'esecuzione della pipeline di Record Linkage.

Usage:
    python run_pipeline.py --step all         # Esegue tutto
    python run_pipeline.py --step download    # Solo download datasets
    python run_pipeline.py --step schema      # Solo schema mediato
    python run_pipeline.py --step ground_truth  # Solo ground truth
    python run_pipeline.py --step train       # Solo training modello
"""

import argparse
import subprocess
import sys
import os


def run_step(script_path: str, args: list = None):
    """Esegue uno script Python con argomenti opzionali."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Esecuzione: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=os.path.dirname(script_path) or '.')
    
    if result.returncode != 0:
        print(f"❌ Errore nell'esecuzione di {script_path}")
        sys.exit(result.returncode)
    
    print(f"✅ Completato: {os.path.basename(script_path)}")


def main():
    parser = argparse.ArgumentParser(description='Pipeline Record Linkage - Entry Point')
    parser.add_argument('--step', type=str, required=True,
                        choices=['all', 'download', 'schema', 'ground_truth', 'train'],
                        help='Step da eseguire')
    parser.add_argument('--base-path', type=str, default='.',
                        help='Path base del progetto')
    args = parser.parse_args()
    
    base_path = os.path.abspath(args.base_path)
    
    # Paths degli script (nuova struttura src/)
    download_script = os.path.join(base_path, 'src/data/download.py')
    schema_script = os.path.join(base_path, 'src/schema/mediated_schema.py')
    gt_script = os.path.join(base_path, 'src/schema/ground_truth.py')
    train_script = os.path.join(base_path, 'src/pipelines/record_linkage.py')
    
    print("=" * 60)
    print("RECORD LINKAGE PIPELINE")
    print(f"Base path: {base_path}")
    print(f"Step: {args.step}")
    print("=" * 60)
    
    script_args = ['--base-path', base_path]
    
    if args.step == 'all':
        run_step(download_script, ['--dataset', 'all'])
        run_step(schema_script, script_args)
        run_step(gt_script, script_args)
        run_step(train_script, script_args + ['--train'])
    
    elif args.step == 'download':
        run_step(download_script, ['--dataset', 'all'])
        
    elif args.step == 'schema':
        run_step(schema_script, script_args)
        
    elif args.step == 'ground_truth':
        run_step(gt_script, script_args)
        
    elif args.step == 'train':
        run_step(train_script, script_args + ['--train'])
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETATA!")
    print("=" * 60)


if __name__ == '__main__':
    main()
