# Learning Methodology <!-- omit in toc -->

- [Data Pre-Processing](#data-pre-processing)
  - [Object State Estimations](#object-state-estimations)
  - [Myrmex Processing](#myrmex-processing)
- [Network Trainings](#network-trainings)
  - [Training I: cos($\alpha$) as target](#training-i-cosalpha-as-target)
- [Next Steps](#next-steps)

## Data Pre-Processing

Before training, we analyze myrmex (input) data and labels. 
The goal is to filter out false label detections and to determine how to equalize myrmex data in terms of sequence length.
First, we visualize all rotation samples to get an idea of how much of the space we've covered during data collection (cuboid set).
From visual inspection, it seems like one quadrant is underrepresented.
It will be interesting to test the network's performance on samples collected in that region.

![data samples distribution](./plots/data_dist.png)

### Object State Estimations

For object state estimates, we only consider data points that were measured before our detected table contact $T$, minus a small time constant $\zeta$ (200ms) to account for delay in the contact detection.
Tag detections can be a bit unreliable at times / be subject to jitter. 
Our setup observes markers over a period of time, thus bad samples are likely to occur. 
Additionally, we potentially have multiple detections per time step (max. one per camera) and we need to decide which one to use or how to reduce them to one label per time step.

The camera $c$ reports a quaternion $q^c(t)$ that describes the rotation of the tables $-Z$ axis to the object's current placing normal.
For each sample we can plot the sequence of $q^c(t)$ for each camera and visually verify whether the data makes sense.
Note, that the number of observations per sample must not be equal among cameras.

Analyzing the data in this way, we find two cases of potentially bad samples: outliers and high-variance sequences.
First, plots of samples that contain outliers:

![plot 1 outlier](./plots/outlier_1.png) | ![plot 1 outlier](./plots/outlier_2.png)
-|-

To identify camera sequences with outlier, we can calculate the angle $\alpha^c(t)$ of each estimated object normal to the desired one ($-Z$ in table coordinates).
Then, we calculate the standard deviation of this sequence and if it is higher than a certain threshold, we don't take that camera's measurement into account when producing the final label.

Second, plots of camera sequences with high measurement deviations:

![plot high var 1](./plots/bad_1.png) | ![plot high var 2](.plot/../plots/high_var.png)
-|-

Here, the same metric is effective filter out unreliable cameras readings.
Examining the left plot, we can see that one camera sequence has a high standard deviation, and the only other one only two samples.
Both sequences are not reliable, hence we need to filter such samples.

We arrive at the following filter rules to reject camera sequences if, ...

*  ... they have less than $N_min$ (10) samples
*  ... their standard deviation is above $\sigma_{min}$ (0.005)

From the cuboid dataset, we reject only 2 out of 203 samples, the one we saw above and one where all cameras only measured 9 object states.

![plot bad 1](./plots/bad_1.png) | ![plot bad 2](.plot/../plots/bad_2.png)
-|-

**Conversion into Lables**

*Option I: Dot Product*
The dot product between placing normal and object normal is a first, simple label to test network architectures and training setups in general.
However, the value is ambiguous (vectors on a circle with same radius around the placing normal have the same dot product with the placing normal).

*Option II: Polar Coordinates*
Spherical coordinates give us the angle for both rotation required. 
Better use those, since from the angles we can reconstruct the full rotation.

To get polar coordinates (the two angles), we calculate the angle $\theta$ between the table's normal and the placing face's normal in the -Y/Z plane and the angle $\phi$ between the Y-axis and the placing face normal in the X/Z plane.
Rotating by $\theta$ about the X-axis and then by $\phi$ about the Z-axis (stationary base frame axes), we arrive at the object's orientation.

![polar coordinate conversion](./plots/polar_coords.png)

Here, $\mathbf v$ is the label and $\mathbf u$ the result of rotating $-Z$ as described above. 
The vectors don't match exactly, and $v \dot u = 0.995$.
Not sure if this is sufficient or not.

### Myrmex Processing

The myremx data needs to be of equal sequence length (at least right and left tactile sequences of the same sample).
To do so, we can take a look at the mean sensor activation (i.e. averaged over all cells) per time step and see how many samples of a sequence actually measure an object state change.

![mm sequences](./plots/mm_lookback.png) | ![mm sequences mean](./plots/mm_lookback_avg.png)
-|-

Here, we just choose a look back window (50 samples) that includes all deviations, including some samples that measure the object without contact.
This can be seen by the vertical blue line in the plots.

A visualization of an example sequence looks like this:

![mm gif](.plots/../plots/myrmex.gif)

## Network Trainings

### Training I: cos($\alpha$) as target

* 80/20 train-test split on cuboid dataset
* TactileInsertionNet
* 1 episode (= 20 batches)
* MSE Loss
* label were dot product pf $-Z$ and object's placing face normal
* batch size of 8
* also test performance on the cylinder set to test for generalization

![first training with dot product training curve](./plots/training_dot.png)

## Next Steps

**Network Evaluation**

* training with polar coordinates as targets
* visualization of predictions (plot vectors)
* network test on robot (get sample, predict, plot)

**Paper**

* make overleaf template
* start with introduction
* related work section 
  * search for more placing literature
  * add insertion net etc.
* describe experiment setup

**Experiments**

* RL setup 
* more objects? test objects (real life)?