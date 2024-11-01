#!/bin/bash
# -*- coding: utf-8 -*-

# translate & post-process
python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.0003_batch500/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.0003_batch500.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.0003_batch500.txt assignments/03/baseline/translations_lr0.0003_batch500.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.0003_batch1000/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.0003_batch1000.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.0003_batch1000.txt assignments/03/baseline/translations_lr0.0003_batch1000.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.0003_batch2000/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.0003_batch2000.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.0003_batch2000.txt assignments/03/baseline/translations_lr0.0003_batch2000.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.001_batch500/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.001_batch500.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.001_batch500.txt assignments/03/baseline/translations_lr0.001_batch500.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.001_batch1000/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.001_batch1000.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.001_batch1000.txt assignments/03/baseline/translations_lr0.001_batch1000.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.001_batch2000/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.001_batch2000.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.001_batch2000.txt assignments/03/baseline/translations_lr0.001_batch2000.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.003_batch500/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.003_batch500.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.003_batch500.txt assignments/03/baseline/translations_lr0.003_batch500.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.003_batch1000/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.003_batch1000.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.003_batch1000.txt assignments/03/baseline/translations_lr0.003_batch1000.p.txt en

python translate.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints_lr0.003_batch2000/checkpoint_best.pt --output assignments/03/baseline/translations_lr0.003_batch2000.txt
bash scripts/postprocess.sh assignments/03/baseline/translations_lr0.003_batch2000.txt assignments/03/baseline/translations_lr0.003_batch2000.p.txt en

# BLEU score
cat assignments/03/baseline/translations_lr0.0003_batch500.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.0003_batch1000.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.0003_batch2000.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.001_batch500.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.001_batch1000.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.001_batch2000.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.003_batch500.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.003_batch1000.p.txt | sacrebleu data/en-fr/raw/test.en
cat assignments/03/baseline/translations_lr0.003_batch2000.p.txt | sacrebleu data/en-fr/raw/test.en

