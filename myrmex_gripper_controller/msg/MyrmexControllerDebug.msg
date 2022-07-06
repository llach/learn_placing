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
float32 f_sum
float32 f_target

float32[] f
float32[] bias
float32[] f_thresholds

float32 Ki
float32 Kp
float32 delta_q
float32 delta_f
float32 max_error
float32 error_integral

float32[] u
float32[] q_T
float32[] des_q
