#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "TIAGo hostname or IP missing!"
    exit 1
fi

ssh -t $1 << EOF
source /opt/pal/ferrum/setup.bash
rosservice call /controller_manager/load_controller "name: 'myrmex_gripper_controller'"
rosservice call /controller_manager/switch_controller "start_controllers:
- 'myrmex_gripper_controller'
stop_controllers:
- 'gripper_controller'
- 'gripper_force_controller'
strictness: 0"
EOF