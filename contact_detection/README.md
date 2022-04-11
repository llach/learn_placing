# Force/Torque Sensor based table contact detection

Subscribes to `/wrist_ft` topic (published at ~50Hz) and publishes on `/table_contact/in_contact` a bool indicating whether we're in table contact or not.

## Procedure

1. Calibration phase:
   1. collect $N$ raw wrench samples from the FT sensor
   2. calculate means $\mu_i$ and standard deviations $\sigma_i$ from these samples for each wrench **force** component $i \in \{x, y, z\}$ (e.g. $\mu_x$ for the force along the x axis)
2. Contact detection phase:
   1. calculate each component's deviation from the mean $\Delta w_i(t) =  |f_i(t)-\mu_i|$
   2. determine whether a component's deviation is high: $c_i(t) = 1$ if $\Delta w_i(t) > \gamma \sigma_i$ else $c_i(t) = 0$, where $\gamma$ is some scaling factor 
   3. store the last $M$ in-contact boolean values $c_i$, that is $C_i = \{c_i(t-M), ..., c_i(t) \}$ 
   4. report table contact if $med(C_i) = 1$, where $med(x)$ is the median operator

### Parameters

* $N$ - number of sensor samples (150)
* $M$ - size of the set of in-contact boolean values (5)
* $\gamma$ - $\gamma \sigma_i$ is the threshold for contact detection (8)