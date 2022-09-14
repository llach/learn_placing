* optitrack is not perfect, sometimes rotation wabbles a bit
  * also sometimes markers can be occluded, thus not perfect
  * in very few cases, metric can be incorrect due to old transforms
  * -> this can also lead to the ang. offset metric looking worse than it is
* FT only seem to try to straighten the gripper
  * sometimes that works
  * good example is the last pose where the object is nearly perfect but the sensor is nearly orthogonal to the table
* cylinder is tricky to place since it 
  * round
  * small surface area
  * top heavy
* for cuboid objects (e.g. Salt), larger errors can still result in good placements
* there were multiple robot restarts / days between data recordings and also in the evaluation
* tabasco was difficult, due to its center of mass
* which ang offset is good for placing is object dependent (placing surface size and COM)
* tactile&FT underperforms on Toothpast, but it was often very very close. the COM makes this a difficult object