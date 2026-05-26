#!/usr/bin/env python3
import signal
import sys
import time
from ServerWrapper import ServerWrapper

num_robots = 1
wrapper = ServerWrapper(num_robots)

def signal_handler(sig, frame):
    wrapper.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Wait for robot to connect
print("Waiting for robot...")
while wrapper.get_num_active() < num_robots:
    time.sleep(0.1)
print("Robot connected. Driving forward for 3 seconds...")

# Drive forward
wrapper.set_velocities([[0.05, 0.0]])
wrapper.step(rate=10, time=3000)

wrapper.stop()
print("Done.")
