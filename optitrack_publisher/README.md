# optitrack_publisher

1. Subscribe to all trackers published via VRPN (updated peridodically)
2. Transform poses from left-handed to right-handed frames (OptiTrack uses the weird convention)
3. Advertise fixed TIAGo -> TIAGo_opti transform

**Got TF time delay issues?** try running `ntpdate TIAGO_IP` on the laptop