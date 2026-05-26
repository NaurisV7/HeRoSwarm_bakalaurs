#!/usr/bin/env python3
import signal
import sys
import numpy as np
from ServerWrapper import *
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

num_robots = 3

wrapper = ServerWrapper(num_robots)

def signal_handler(sig, frame):
    wrapper.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

L = completeGL(num_robots)

while True:
    try:
        current_pos = wrapper.get_data("global_pos")
        if len(current_pos) < num_robots:
            continue

        robot_light_sensor_data = wrapper.get_light()
        current_pos_xy = [x[:2] for x in current_pos]

        vels = []
        for i in range(num_robots):
            j = topological_neighbors(L, i)

            intensities = []
            for x in j:
                light = robot_light_sensor_data[x]
                intensities.append(sum(light[:3]) if light is not None else 0)

            max_idx = intensities.index(max(intensities))
            target = j[max_idx]
            vels.append([
                current_pos_xy[target][0] - current_pos_xy[i][0],
                current_pos_xy[target][1] - current_pos_xy[i][1]
            ])

        vels = si_barrier_cert(
            np.asarray(vels).transpose(),
            np.asarray(current_pos_xy).transpose()
        )
        vels = si_to_uni_dyn(vels, np.asarray(current_pos).transpose())

        wrapper.set_velocities(vels.transpose())
        wrapper.step(rate=10, time=500)

    except Exception as e:
        print(e)
        wrapper.stop()
        break

wrapper.stop()
