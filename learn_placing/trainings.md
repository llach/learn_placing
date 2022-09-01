
## Experiment Variables

* inputs modalities:
  * gripper pose (proprioception)
  * tactile data
  * force/torque
* input data variation / active touch:
  * static
  * dynamic
* datasets:
  * object-to-gripper variation (O2G)
  * gripper-to-world variation (G2W)

## Questions & Hypotheses

1. How important is the tap (=active touch)?
   * H: in the G2W case, only the tap will contain information about object state
   * Exp: G2W dataset, without proprioception: compare static inputs vs active touch -> object pose shouldn't be predictable from static input only
2. Which combination of input modalities is necessary to predict the object's state?
   * Exp: train models that contain all combination of input modalities
     * [tactile, ft, prop.] on their own
     * [tact, ft], [tact, prop.], [ft, prop.] 
     * [tactile, ft, prop.] all together
3. How good do these networks generalize to unknown objects?

## TODOs

* ~~add ft information~~
* make network configurable for all possible input modalities
* train for all input modality variations
  
* analyze static vs with tap on new datasets
  
* robot evaluation setup:
  * generating plan for pose delta (on TIAGo++)
  * vision baseline: correcting pose based on vision / (noisy?) ground truth data
  * collect and execute sample on robot

* collect larger dataset with all variations combined
* test for generalisation