#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "TIAGo hostname or IP missing!"
    exit 1
fi

ssh -t $1 << EOF
mkdir -p /home/pal/.pal/pal_startup/apps
mkdir -p /home/pal/.pal/pal_startup/control
EOF

scp ./start_contact_detection.yaml $1:/home/pal/.pal/pal_startup/apps/
scp ./ft_contact_detection.yaml $1:/home/pal/.pal/pal_startup/control/