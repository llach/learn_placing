#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "TIAGo hostname or IP missing!"
    exit 1
fi

ssh -t $1 << EOF
source /opt/pal/ferrum/setup.bash
rosservice call /controller_manager/load_controller "name: 'torso_stop_controller'"
rosservice call /controller_manager/switch_controller "start_controllers:
- 'torso_stop_controller'
stop_controllers:
- 'torso_controller'
strictness: 0"
EOF