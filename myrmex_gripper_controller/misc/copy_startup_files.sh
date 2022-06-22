#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "TIAGo hostname or IP missing!"
    exit 1
fi

ssh -t $1 << EOF
mkdir -p /home/pal/.pal/pal_startup/apps
mkdir -p /home/pal/.pal/pal_startup/control
EOF

scp ./load_myrmex_gripper_controller.yaml $1:/home/pal/.pal/pal_startup/apps/
scp ./myrmex_controller.yaml $1:/home/pal/.pal/pal_startup/control/