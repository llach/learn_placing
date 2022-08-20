
## Tactile-RL for Insertion
https://arxiv.org/pdf/2104.01167.pdf

* 2x GelSim sensors (640x480 px)
* their contact duration are 0.5s
* during contact, they record 30 images per sensor, which they downsample to 12 each to combat vanishing gradients in the RNNs
* F/T sensor data (6x1) was downsampled from 120 to 32 samples
* network bootstrapping: supervised pre-training with 300 samples
* during first 50 episodes of RL trainings, actor was frozen
* 50 samples (25 episodes) for new objects
* 500 samples (150 episodes) to adapt to U-shape environment
* 2000 samples (300 episodes) for hole environment
* training: 3000 samples, 8 hours in total
* 2000 evaluation runs: 250 trials for 4 known and 4 unknown objects

network arch: 

(2, 12, 640, 480) [N_SENSORS, N_IMAGES, W, H] -> {
    4x CNN with kernels of size [5, 3, 3, 3], extracts 1x512 features
    2x LSTM with 512 neurons
} [for each GelSim]
2x fully connected, [512, 256]
output: 1x3 (x, y, theta)

code: https://github.com/siyuandong16/Tactile_insertion_with_RL

## InsertionNet v1
https://arxiv.org/pdf/2104.14223.pdf

* 16 objects
* 100 training samples from backwards learning (not sure if this is per object)
* data augmentation -> 10000 repeats on 64000 sample batches -> 640000 samples in total
* 200 test iterations per object -> 3200 test repeats

## InsertionNet v2 
https://arxiv.org/pdf/2203.01153.pdf

* 16 objects
* 150 training samples from backwards learning (not sure if this is per object)
* data augmentation -> 1500 repeats on 64 sample batches -> 192000 samples in total
* 200 test iterations per object -> 3200 test repeats