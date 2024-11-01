#!/bin/bash
# -*- coding: utf-8 -*-

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.0003_batch500 --lr 0.0003 --batch-size 500

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.0003_batch1000 --lr 0.0003 --batch-size 1000

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.0003_batch2000 --lr 0.0003 --batch-size 2000

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.001_batch500 --lr 0.001 --batch-size 500

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.001_batch1000 --lr 0.001 --batch-size 1000

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.001_batch2000 --lr 0.001 --batch-size 2000

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.003_batch500 --lr 0.003 --batch-size 500

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.003_batch1000 --lr 0.003 --batch-size 1000

python train.py --data data/en-fr/prepared --source-lang fr --target-lang en --save-dir assignments/03/baseline/checkpoints_lr0.003_batch2000 --lr 0.003 --batch-size 2000