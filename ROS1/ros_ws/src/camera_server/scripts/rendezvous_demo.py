#!/usr/bin/env python3
import signal
import sys
import time
import math
import json
import os
import numpy as np
from itertools import permutations
from ServerWrapper import ServerWrapper

num_robots = 4
v_max = 0.04
omega_max = 0.5
ramp_duration = 3.0
stop_threshold = 0.04
arrival_threshold = 0.20
half_side = 0.09
v_hold = 0.008
repulsion_radius = 0.20
repulsion_gain = 0.08
STALE_TIMEOUT = 0.5
COLLISION_THRESHOLD = 0.10
NEAR_MISS_THRESHOLD = 0.15

wrapper = ServerWrapper(num_robots)

def signal_handler(sig, frame):
    wrapper.stop()
    save_metrics()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("Waiting for {} robots...".format(num_robots))
while wrapper.get_num_active() < num_robots:
    time.sleep(0.1)
print("Robots connected.")

def unicycle_to_point(current, target):
    delta_x = target[0] - current[0]
    delta_y = target[1] - current[1]
    dist = math.sqrt(delta_x**2 + delta_y**2)
    if dist < stop_threshold:
        return 0.0, 0.0
    theta = current[2]
    desired_theta = math.atan2(delta_y, delta_x)
    heading_error = math.atan2(math.sin(desired_theta - theta), math.cos(desired_theta - theta))
    omega = max(-omega_max, min(omega_max, 2.0 * heading_error))
    speed = min(v_max, dist * 0.3)
    v = speed * max(0.0, math.cos(heading_error))
    return v, omega

def best_assignment(positions, targets):
    best_cost = float('inf')
    best_perm = list(range(len(positions)))
    for perm in permutations(range(len(positions))):
        cost = sum(math.sqrt((positions[i][0] - targets[perm[i]][0])**2 +
                             (positions[i][1] - targets[perm[i]][1])**2)
                   for i in range(len(positions)))
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)
    return best_perm

def inter_robot_distances(positions):
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = math.sqrt((positions[i][0] - positions[j][0])**2 +
                          (positions[i][1] - positions[j][1])**2)
            dists.append(d)
    return dists

# Center the square on the initial centroid of all robots
initial_pos = wrapper.get_data("global_pos")
cx, cy = np.mean([p[:2] for p in initial_pos], axis=0)
targets = [
    [cx - half_side, cy - half_side],
    [cx + half_side, cy - half_side],
    [cx + half_side, cy + half_side],
    [cx - half_side, cy + half_side],
]
print("Formation center: ({:.2f}, {:.2f})  Square: {:.0f}cm x {:.0f}cm".format(
    cx, cy, half_side * 200, half_side * 200))

from datetime import datetime
experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

metrics = {
    "experiment_id": "run_" + experiment_timestamp,
    "timestamp": datetime.now().isoformat(),
    "condition": "baseline",  # CHANGE TO: "1_robot_hidden" or "2_robots_hidden"
    "hidden_robot_ids": [],   # CHANGE TO: [2] or [1,3] etc.
    "hide_duration_s": None,  # CHANGE TO: 3.0 or 5.0 etc.
    "config": {
        "num_robots": num_robots,
        "square_side_m": round(half_side * 2, 3),
        "v_max": v_max,
        "omega_max": omega_max,
        "arrival_threshold": arrival_threshold,
        "repulsion_radius": repulsion_radius,
    },
    "result": {
        "completed": False,
        "time_to_complete_s": None,
    },
    "initial": {
        "avg_dist_to_target_m": None,
        "max_dist_to_target_m": None,
        "avg_inter_robot_dist_m": None,
    },
    "final": {
        "avg_dist_to_target_m": None,
        "max_dist_to_target_m": None,
        "avg_inter_robot_dist_m": None,
    },
    "travel": {
        "total_distance_m": 0.0,
        "per_robot_distance_m": [0.0] * num_robots,
    },
    "safety": {
        "collision_count": 0,
        "min_inter_robot_dist_m": float('inf'),
        "near_miss_count": 0,
    },
    "localization": {
        "position_loss_count": 0,
        "avg_recovery_time_s": None,
        "_recovery_times": [[] for _ in range(num_robots)],
        "_loss_count_per_robot": [0] * num_robots,
    },
}

metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
all_runs_file = os.path.join(metrics_dir, "all_runs.json")

def save_metrics():
    if metrics["safety"]["min_inter_robot_dist_m"] == float('inf'):
        metrics["safety"]["min_inter_robot_dist_m"] = None
    all_recoveries = [t for r in metrics["localization"]["_recovery_times"] for t in r]
    metrics["localization"]["avg_recovery_time_s"] = round(sum(all_recoveries) / len(all_recoveries), 2) if all_recoveries else None
    metrics["localization"]["position_loss_count"] = sum(metrics["localization"]["_loss_count_per_robot"])
    del metrics["localization"]["_recovery_times"]
    del metrics["localization"]["_loss_count_per_robot"]
    metrics["travel"]["total_distance_m"] = round(metrics["travel"]["total_distance_m"], 4)
    metrics["travel"]["per_robot_distance_m"] = [round(d, 4) for d in metrics["travel"]["per_robot_distance_m"]]

    # Append to all_runs.json
    runs = []
    if os.path.exists(all_runs_file):
        with open(all_runs_file, 'r') as f:
            try:
                runs = json.load(f)
            except Exception:
                runs = []
    runs.append(metrics)
    with open(all_runs_file, 'w') as f:
        json.dump(runs, f, indent=2)

    # Also save/update CSV for easy Excel import
    csv_file = os.path.join(metrics_dir, "all_runs.csv")
    import csv
    write_header = not os.path.exists(csv_file)
    flat = {
        "experiment_id": metrics["experiment_id"],
        "timestamp": metrics["timestamp"],
        "condition": metrics["condition"],
        "hidden_robot_ids": str(metrics["hidden_robot_ids"]),
        "hide_duration_s": metrics["hide_duration_s"],
        "num_robots": metrics["config"]["num_robots"],
        "square_side_m": metrics["config"]["square_side_m"],
        "completed": metrics["result"]["completed"],
        "time_to_complete_s": metrics["result"]["time_to_complete_s"],
        "initial_avg_dist_m": metrics["initial"]["avg_dist_to_target_m"],
        "initial_max_dist_m": metrics["initial"]["max_dist_to_target_m"],
        "initial_avg_inter_robot_m": metrics["initial"]["avg_inter_robot_dist_m"],
        "final_avg_dist_m": metrics["final"]["avg_dist_to_target_m"],
        "final_max_dist_m": metrics["final"]["max_dist_to_target_m"],
        "final_avg_inter_robot_m": metrics["final"].get("avg_inter_robot_dist_m"),
        "total_distance_m": metrics["travel"]["total_distance_m"],
        "robot0_distance_m": metrics["travel"]["per_robot_distance_m"][0],
        "robot1_distance_m": metrics["travel"]["per_robot_distance_m"][1],
        "robot2_distance_m": metrics["travel"]["per_robot_distance_m"][2],
        "robot3_distance_m": metrics["travel"]["per_robot_distance_m"][3],
        "collision_count": metrics["safety"]["collision_count"],
        "min_inter_robot_dist_m": metrics["safety"]["min_inter_robot_dist_m"],
        "near_miss_count": metrics["safety"]["near_miss_count"],
        "position_loss_count": metrics["localization"]["position_loss_count"],
        "avg_recovery_time_s": metrics["localization"]["avg_recovery_time_s"],
    }
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(flat)

    print("Run #{} saved to: {}".format(len(runs), all_runs_file))
    print("CSV updated: {}".format(csv_file))

assignment = None
start_time = time.time()
prev_positions = None
metrics_initialized = False
loop_count = 0
completion_reported = False

robot_was_stale = [False] * num_robots
stale_start_time = [None] * num_robots
was_in_collision = False

for _ in range(2000):
    try:
        current_pos = wrapper.get_data("global_pos")
        pos_times = wrapper.get_data("global_pos_time")
        if len(current_pos) < num_robots:
            wrapper.step(rate=10, time=100)
            continue

        now = time.time()

        if assignment is None:
            assignment = best_assignment(current_pos, targets)
            metrics["assignment"] = assignment
            print("Assigned: robot {} -> corners {}".format(list(range(num_robots)), assignment))

        dists_to_target = [
            math.sqrt((current_pos[i][0] - targets[assignment[i]][0])**2 +
                      (current_pos[i][1] - targets[assignment[i]][1])**2)
            for i in range(num_robots)
        ]

        ir_dists = inter_robot_distances(current_pos)
        min_ir = min(ir_dists)
        avg_ir = sum(ir_dists) / len(ir_dists)

        if not metrics_initialized:
            metrics["initial"]["avg_dist_to_target_m"] = round(sum(dists_to_target) / num_robots, 4)
            metrics["initial"]["max_dist_to_target_m"] = round(max(dists_to_target), 4)
            metrics["initial"]["avg_inter_robot_dist_m"] = round(avg_ir, 4)
            prev_positions = [[p[0], p[1]] for p in current_pos]
            metrics_initialized = True

        for i in range(num_robots):
            dx = current_pos[i][0] - prev_positions[i][0]
            dy = current_pos[i][1] - prev_positions[i][1]
            step_dist = math.sqrt(dx**2 + dy**2)
            metrics["travel"]["total_distance_m"] += step_dist
            metrics["travel"]["per_robot_distance_m"][i] += step_dist
        prev_positions = [[p[0], p[1]] for p in current_pos]

        in_collision = min_ir < COLLISION_THRESHOLD
        if in_collision and not was_in_collision:
            metrics["safety"]["collision_count"] += 1
            print("COLLISION #{} detected! Pair distance: {:.3f}m".format(
                metrics["safety"]["collision_count"], min_ir))
        was_in_collision = in_collision

        if min_ir < metrics["safety"]["min_inter_robot_dist_m"]:
            metrics["safety"]["min_inter_robot_dist_m"] = round(min_ir, 4)
        if min_ir < NEAR_MISS_THRESHOLD:
            metrics["safety"]["near_miss_count"] += 1

        vels = []
        all_arrived = True
        for i in range(num_robots):
            curr = current_pos[i]
            target = targets[assignment[i]]
            is_stale = (now - pos_times[i] > STALE_TIMEOUT)

            if is_stale and not robot_was_stale[i]:
                metrics["localization"]["_loss_count_per_robot"][i] += 1
                stale_start_time[i] = now
                print("Robot {} lost localization (loss #{})".format(
                    i, metrics["localization"]["_loss_count_per_robot"][i]))

            if not is_stale and robot_was_stale[i] and stale_start_time[i] is not None:
                recovery_duration = round(now - stale_start_time[i], 2)
                metrics["localization"]["_recovery_times"][i].append(recovery_duration)
                stale_start_time[i] = None
                print("Robot {} recovered in {:.1f}s".format(i, recovery_duration))

            robot_was_stale[i] = is_stale

            if is_stale:
                vels.append([0.0, 0.0])
                all_arrived = False
                continue

            corner_dist = dists_to_target[i]
            if corner_dist > arrival_threshold:
                all_arrived = False

            if completion_reported:
                # Holding mode: creep toward corner at fixed very slow speed
                delta_x = target[0] - curr[0]
                delta_y = target[1] - curr[1]
                desired_theta = math.atan2(delta_y, delta_x)
                heading_error = math.atan2(math.sin(desired_theta - curr[2]),
                                           math.cos(desired_theta - curr[2]))
                omega = max(-omega_max, min(omega_max, 2.0 * heading_error))
                v = v_hold * max(0.0, math.cos(heading_error))
                vels.append([v, omega])
                continue

            # Speed scale 1: slow down based on distance to actual corner
            ramp = min(1.0, (now - start_time) / ramp_duration)
            corner_scale = max(0.15, min(1.0, corner_dist / 0.25)) * ramp + 0.20 * (1 - ramp)

            # Speed scale 2: slow down when neighbors are close
            min_neighbor = min(
                math.sqrt((curr[0] - current_pos[j][0])**2 + (curr[1] - current_pos[j][1])**2)
                for j in range(num_robots) if j != i)
            neighbor_scale = max(0.20, min(1.0, (min_neighbor - 0.15) / 0.25))

            speed_scale = corner_scale * neighbor_scale

            # Repulsion only active when robot is far from its corner (> 20cm)
            eff_x, eff_y = target[0], target[1]
            if corner_dist > 0.20:
                for j in range(num_robots):
                    if j == i:
                        continue
                    ddx = curr[0] - current_pos[j][0]
                    ddy = curr[1] - current_pos[j][1]
                    d = math.sqrt(ddx**2 + ddy**2)
                    if d < repulsion_radius and d > 0.001:
                        strength = repulsion_gain * (1 - d / repulsion_radius)
                        eff_x += strength * ddx / d
                        eff_y += strength * ddy / d

            v, omega = unicycle_to_point(curr, [eff_x, eff_y])
            vels.append([v * speed_scale, omega])

        if all_arrived and not completion_reported:
            elapsed = now - start_time
            metrics["result"]["completed"] = True
            metrics["result"]["time_to_complete_s"] = round(elapsed, 2)
            metrics["final"]["avg_dist_to_target_m"] = round(sum(dists_to_target) / num_robots, 4)
            metrics["final"]["max_dist_to_target_m"] = round(max(dists_to_target), 4)
            metrics["final"]["avg_inter_robot_dist_m"] = round(avg_ir, 4)
            completion_reported = True
            print("=== Rendezvous complete! Time: {:.1f}s === Holding formation...".format(elapsed))
            save_metrics()

        print("---")
        for i in range(num_robots):
            print("Robot {}: dist={:.3f}m".format(i, dists_to_target[i]))

        wrapper.set_velocities(vels)
        wrapper.step(rate=10, time=100)
        loop_count += 1

    except Exception as e:
        print(e)
        break

wrapper.stop()
if not completion_reported:
    save_metrics()
print("Done.")
