import os
import bagpy
from bagpy import bagreader

import pandas as pd

def ros1_check_file(bag_path,required_topics):

    if type(bag_path) == str or type(bag_path) == list and len(bag_path) == 1:  
        
        if type(bag_path) == list:
            bag_path = bag_path[0]
        
        print(f"\nChecking topics in: {bag_path}")
        if not os.path.exists(bag_path):
            print(" Bag file does not exist.")
            return False
        try:
            b = bagreader(bag_path)
            topics = b.topics
            print("Topics found:", topics)
            for topic in required_topics:
                if topic in topics:
                    print(f" {topic} found")
                else:
                    print(f" {topic} NOT found")
            return True
        
        except Exception as e:
            print(f"Error reading {bag_path}: {e}")
            return False
        
    elif type(bag_path) == list:
        for path in bag_path:
            if not ros1_check_file(path, required_topics):
                return False
        return True
    else:
        print("Invalid bag_path type.")
        return False
    

def ros1_load_topic(bag_path, topic_name):
    if not os.path.exists(bag_path):
        print(" Bag file does not exist.")
        return None
    try:
        b = bagreader(bag_path)
        if topic_name in b.topics:
            csvfile = b.message_by_topic(topic_name)
            df = pd.read_csv(csvfile)
            return df
        else:
            print(f"Topic {topic_name} not found in {bag_path}.")
            return None
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        return None

# for ros2 
# """
# load_data.py
# ────────────────────────────────────────────────────────────
# Helper module for the Go2 ROS2 data collection notebook.
# Contains all kinematics, config loading, and MCAP utilities
# so the notebook stays clean.
# """

# import os
# import numpy as np
# from mcap.reader import make_reader
# from mcap_ros2.decoder import DecoderFactory


# # ════════════════════════════════════════════════════════════
# #  CONFIG
# # ════════════════════════════════════════════════════════════

# def load_config():
#     """
#     Returns a dict of {mcap_path: label} for all known bag files.
#     Edit this list to add new recordings.
#     Supports .mcap and the custom .emav extension (same format).
#     """
#     return {
#         "./data/dataset_path": "stiff_run",
#         # Add more paths below — .mcap or .emav both work
#         # "./data/dataset_path": "terrain_label",
#     }


# # ════════════════════════════════════════════════════════════
# #  MCAP UTILITIES
# # ════════════════════════════════════════════════════════════

# def get_topics(mcap_path: str) -> set:
#     """Return the set of all topic names in an mcap/emav file."""
#     with open(mcap_path, "rb") as f:
#         reader = make_reader(f, decoder_factories=[DecoderFactory()])
#         summary = reader.get_summary()
#         return {ch.topic for ch in summary.channels.values()}


# def get_schemas(mcap_path: str) -> dict:
#     """Return {schema_name: schema_text} for all schemas in the file."""
#     with open(mcap_path, "rb") as f:
#         reader = make_reader(f, decoder_factories=[DecoderFactory()])
#         summary = reader.get_summary()
#         result = {}
#         for schema in summary.schemas.values():
#             text = schema.data.decode("utf-8", errors="ignore") if schema.data else ""
#             result[schema.name] = text
#         return result


# def sample_messages(mcap_path: str, topic: str, n: int = 3) -> list:
#     """Return up to n decoded messages from a given topic."""
#     msgs = []
#     with open(mcap_path, "rb") as f:
#         reader = make_reader(f, decoder_factories=[DecoderFactory()])
#         for _, channel, _, ros_msg in reader.iter_decoded_messages(topics=[topic]):
#             msgs.append(ros_msg)
#             if len(msgs) >= n:
#                 break
#     return msgs


# def check_required_topics(mcap_path: str, required: list) -> dict:
#     """
#     Check which required topics are present.
#     Returns {topic: True/False}.
#     """
#     found = get_topics(mcap_path)
#     return {t: (t in found) for t in required}


# # ════════════════════════════════════════════════════════════
# #  GO2 KINEMATICS
# # ════════════════════════════════════════════════════════════

# HIP_OFFSETS = {
#     "FR": np.array([ 0.1934, -0.0465, 0.0]),
#     "FL": np.array([ 0.1934,  0.0465, 0.0]),
#     "RR": np.array([-0.1934, -0.0465, 0.0]),
#     "RL": np.array([-0.1934,  0.0465, 0.0]),
# }
# L_THIGH      = 0.213
# L_CALF       = 0.213
# L_HIP        = 0.0955
# SIDE         = {"FR": -1, "FL": 1, "RR": -1, "RL": 1}
# JOINTS       = {"FR": (0,1,2), "FL": (3,4,5), "RR": (6,7,8), "RL": (9,10,11)}
# LEGS         = ["FR", "FL", "RR", "RL"]
# JOINT_REORDER = [6, 7, 8, 0, 1, 2, 9, 10, 11, 3, 4, 5]   # Go2 → model [RH,RF,LH,LF]
# CONTACT_THRESH = 20


# def fk(hip_q, thigh_q, calf_q, leg):
#     """Forward kinematics: returns EE position in body frame for one leg."""
#     s      = SIDE[leg]
#     offset = HIP_OFFSETS[leg]
#     p_hip   = np.array([0.0,  s*L_HIP*np.cos(hip_q),          s*L_HIP*np.sin(hip_q)])
#     p_thigh = np.array([0.0, -L_THIGH*np.sin(thigh_q),        -L_THIGH*np.cos(thigh_q)])
#     p_calf  = np.array([0.0, -L_CALF*np.sin(thigh_q+calf_q),  -L_CALF*np.cos(thigh_q+calf_q)])
#     return offset + p_hip + p_thigh + p_calf


# # ════════════════════════════════════════════════════════════
# #  FULL DATA EXTRACTOR
# # ════════════════════════════════════════════════════════════

# def extract_lowstate(mcap_path: str) -> "pd.DataFrame":
#     """
#     Read /lowstate from an mcap/emav file and return a tidy DataFrame with:
#       - time_s         : elapsed time in seconds (starts at 0)
#       - IMU            : acc_xyz, gyro_xyz, roll/pitch/yaw, quaternion
#       - Joints         : q/dq/tau for all 12 joints
#       - EE kinematics  : ee_pos/vel for all 4 legs (body frame, via FK)
#       - Contact        : raw foot_force + binary contact per leg
#     Also prints the detected publish rate.
#     """
#     import pandas as pd

#     rows      = []
#     prev_ee   = {leg: None for leg in LEGS}
#     prev_tick = None
#     tick_diffs = []

#     with open(mcap_path, "rb") as f:
#         reader = make_reader(f, decoder_factories=[DecoderFactory()])
#         for _, channel, _, ros_msg in reader.iter_decoded_messages(topics=["/lowstate"]):
#             ms   = ros_msg.motor_state
#             imu  = ros_msg.imu_state
#             tick = ros_msg.tick

#             if prev_tick is not None:
#                 tick_diffs.append(tick - prev_tick)
#             dt = (tick - prev_tick) / 1000.0 if prev_tick is not None else None

#             row = {
#                 "tick":   tick,
#                 "time_s": tick / 1000.0,
#                 # IMU
#                 "acc_x": imu.accelerometer[0], "acc_y": imu.accelerometer[1], "acc_z": imu.accelerometer[2],
#                 "gyro_x": imu.gyroscope[0],    "gyro_y": imu.gyroscope[1],    "gyro_z": imu.gyroscope[2],
#                 "roll":  imu.rpy[0], "pitch": imu.rpy[1], "yaw": imu.rpy[2],
#                 "quat_x": imu.quaternion[0], "quat_y": imu.quaternion[1],
#                 "quat_z": imu.quaternion[2], "quat_w": imu.quaternion[3],
#                 # Contact
#                 "foot_force_FR": ros_msg.foot_force[0], "foot_force_FL": ros_msg.foot_force[1],
#                 "foot_force_RR": ros_msg.foot_force[2], "foot_force_RL": ros_msg.foot_force[3],
#                 "contact_FR": int(ros_msg.foot_force[0] > CONTACT_THRESH),
#                 "contact_FL": int(ros_msg.foot_force[1] > CONTACT_THRESH),
#                 "contact_RR": int(ros_msg.foot_force[2] > CONTACT_THRESH),
#                 "contact_RL": int(ros_msg.foot_force[3] > CONTACT_THRESH),
#             }

#             for leg, (hi, ti, ci) in JOINTS.items():
#                 row.update({
#                     f"q_{leg}_hip":     ms[hi].q,   f"q_{leg}_thigh":  ms[ti].q,   f"q_{leg}_calf":  ms[ci].q,
#                     f"dq_{leg}_hip":    ms[hi].dq,  f"dq_{leg}_thigh": ms[ti].dq,  f"dq_{leg}_calf": ms[ci].dq,
#                     f"tau_{leg}_hip":   ms[hi].tau_est, f"tau_{leg}_thigh": ms[ti].tau_est, f"tau_{leg}_calf": ms[ci].tau_est,
#                 })
#                 ee_pos = fk(ms[hi].q, ms[ti].q, ms[ci].q, leg)
#                 ee_vel = (ee_pos - prev_ee[leg]) / dt if (prev_ee[leg] is not None and dt and dt > 0) else np.zeros(3)
#                 row.update({
#                     f"ee_pos_{leg}_x": ee_pos[0], f"ee_pos_{leg}_y": ee_pos[1], f"ee_pos_{leg}_z": ee_pos[2],
#                     f"ee_vel_{leg}_x": ee_vel[0], f"ee_vel_{leg}_y": ee_vel[1], f"ee_vel_{leg}_z": ee_vel[2],
#                 })
#                 prev_ee[leg] = ee_pos

#             prev_tick = tick
#             rows.append(row)

#     df = pd.DataFrame(rows)
#     df["time_s"] -= df["time_s"].iloc[0]
#     df["tick"]   -= df["tick"].iloc[0]

#     # Hz report
#     if tick_diffs:
#         median_diff = float(np.median(tick_diffs))
#         hz = 1000.0 / median_diff if median_diff > 0 else 0
#         label = "500 Hz ✓" if abs(hz-500) < 10 else ("100 Hz ✓" if abs(hz-100) < 10 else f"{hz:.1f} Hz ⚠ unexpected")
#         print(f"  Publish rate : {label}  (median tick diff = {median_diff:.1f})")
#         print(f"  Messages     : {len(df)}   Duration: {df['time_s'].iloc[-1]:.2f} s")

#     return df


# def build_npy_arrays(df: "pd.DataFrame") -> dict:
#     """
#     Convert a DataFrame from extract_lowstate() into numpy arrays
#     ready for the deep-contact-estimator model.
#     Returns dict with keys: q, qd, p, v, acc, omega, contacts, contacts_labels, preprocessed_data
#     """
#     q_raw  = df[[f"q_{leg}_{j}"  for leg in LEGS for j in ["hip","thigh","calf"]]].to_numpy()
#     qd_raw = df[[f"dq_{leg}_{j}" for leg in LEGS for j in ["hip","thigh","calf"]]].to_numpy()
#     p_raw  = df[[f"ee_pos_{leg}_{ax}" for leg in LEGS for ax in "xyz"]].to_numpy()
#     v_raw  = df[[f"ee_vel_{leg}_{ax}" for leg in LEGS for ax in "xyz"]].to_numpy()
#     acc    = df[["acc_x",  "acc_y",  "acc_z" ]].to_numpy()
#     omega  = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
#     c_raw  = df[["contact_FR","contact_FL","contact_RR","contact_RL"]].to_numpy().astype(int)

#     q  = q_raw[:,  JOINT_REORDER]
#     qd = qd_raw[:, JOINT_REORDER]
#     p  = p_raw[:,  JOINT_REORDER]
#     v  = v_raw[:,  JOINT_REORDER]

#     reordered_c    = c_raw[:, [2, 0, 3, 1]]          # → [RR, FR, RL, FL]
#     decimal_labels = np.dot(reordered_c, [8, 4, 2, 1])
#     preprocessed   = np.concatenate([q, qd, p, v, acc, omega], axis=1)   # N x 54

#     return {
#         "q": q, "qd": qd, "p": p, "v": v,
#         "acc": acc, "omega": omega,
#         "contacts": c_raw,
#         "contacts_labels": decimal_labels,
#         "preprocessed_data": preprocessed,
#     }


# def save_npy(arrays: dict, output_dir: str, base_name: str):
#     """Save all arrays to output_dir/base_name/*.npy"""
#     save_dir = os.path.join(output_dir, base_name)
#     os.makedirs(save_dir, exist_ok=True)
#     for name, arr in arrays.items():
#         path = os.path.join(save_dir, f"{name}.npy")
#         np.save(path, arr)
#         print(f"  ✓ {name}.npy  {arr.shape}")
#     return save_dir
    
