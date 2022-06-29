int8 TRAJECTORY_EXEC=0
int8 TRANSITION=0
int8 FORCE_CTRL=0

int8 NO_CONTACT=0
int8 LOST_CONTACT=1
int8 GOT_CONTACT=2 
int8 IN_CONTACT=3 
int8 GOAL=4 
int8 VIOLATED=5

std_msgs/Header header

uint32 k
uint32 f_sum
uint32 f_target

uint32[] f
uint32[] f_thresholds

float64 Ki
float64 Kp
float64 delta_q
float64 delta_f
float64 max_error
float64 error_integral

float64[] u
float64[] des_q
