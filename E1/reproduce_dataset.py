#!/usr/bin/env python3
"""Reload a verified E1 dataset/model pair; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset',choices=['all']+list(manifest['datasets']))
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
args=parser.parse_args()
if args.dataset=='all':
    for name in manifest['datasets']:
        print('\n=== '+name+' ===',flush=True)
        subprocess.run([sys.executable,__file__,name,'--device',args.device],check=True)
    sys.exit(0)
entry=manifest['datasets'][args.dataset]
for relative,record in entry['artifacts'].items():
    h=hashlib.sha256()
    with (root/relative).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    if h.hexdigest()!=record['sha256']:raise RuntimeError('Checksum mismatch: '+relative)
art=root/'artifacts'/args.dataset
subprocess.run([sys.executable,str(root/'reproduce.py'),
 '--output_model_path',str(art/'finetuned_model.bin'),
 '--test_path',str(art/'test_dataset.tsv'),
 '--config_path',str(art/'model_config.json'),
 '--vocab_path',str(root.parent/'vocab/hex_vocab.txt'),
 '--labels_num',str(entry['configuration']['labels']),
 '--seq_length','128','--batch_size','32','--pooling','max','--embedding','word',
 '--device',args.device,'--results_dir',str(root/'results'/args.dataset/'reproduced')],
 cwd=root.parent,check=True,env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
