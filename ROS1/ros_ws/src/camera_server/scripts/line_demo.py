#!/usr/bin/env python3
import signal
import sys
import time
import numpy as np
from ServerWrapper import ServerWrapper

num_robots = 1
wrapper = ServerWrapper(num_robots)

def signal_handler(sig, frame):
    wrapper.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Wait for valid position readings for all robots
time.sleep(1.0)
current_pos = wrapper.get_data("global_pos")
while len(current_pos) < num_robots:
    time.sleep(0.1)
    current_pos = wrapper.get_data("global_pos")

# Each robot moves 1m in the x direction from its starting position
targets = []
for i in range(num_robots):
    start = np.array(current_pos[i][:2])
    target = start + np.array([1.0, 0.0])
    targets.append(target)
    print("Robot {}: Start {}  Target {}".format(i, start, target))

wrapper.set_points([[t[0], t[1], 0.0] for t in targets])

stop_threshold = 0.08

for _ in range(500):
    try:
        current_pos = wrapper.get_data("global_pos")
        if len(current_pos) < num_robots:
            wrapper.step(rate=10, time=200)
            continue

        all_reached = True
        for i in range(num_robots):
            dist = float(np.linalg.norm(np.array(current_pos[i][:2]) - targets[i]))
            print("Robot {}: dist {:.3f}".format(i, dist))
            if dist >= stop_threshold:
                all_reached = False

        if all_reached:
            print("All robots reached target!")
            break

        wrapper.step(rate=10, time=200)

    except Exception as e:
        print(e)
        break

wrapper.stop()
print("Done")
