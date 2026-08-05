python train_COG.py -exp COG-1e-4 -t 4 -l 1e-4 -gpu_id cuda:0 &
python train_COG.py -exp COG-1e-3 -t 4 -l 1e-3 -gpu_id cuda:1 &
python train_COG.py -exp COG-le-2 -t 4 -l 1e-2 -gpu_id cuda:2 &
python train_COG.py -exp COG-le-2 -t 4 -l 1e-5 -gpu_id cuda:3 &
python train_COG.py -exp COG-layer8 -t 4 -l 5e-4 -layers 8 -gpu_id cuda:3 &
python train_COG.py -exp COG-layer12 -t 4 -l 5e-4 -layers 12 -gpu_id cuda:2 &

