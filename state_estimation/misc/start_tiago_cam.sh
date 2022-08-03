#!/bin/bash

ssh -t $1 << EOF
screen -d -m "sudo su; roslaunch state_estimation onlyuvc.launch"
EOF
