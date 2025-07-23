import pickle
import numpy as np

file_path = "demo_data/15_demos_2025-07-22_11-37-30.pkl"
video_path = "/home/anker/robotwin/Pi0-RL-RoboTwin/demo_data"
with open(file_path, "rb") as file:
    data = pickle.load(file)
print(data[0]["actions"])
# for transition in data:
#     transition["actions"] = transition["actions"][:, :14]  # Keep only the first 14 dimensions of actions
# # from envs.utils.images_to_video import images_to_video
# # images_to_video(np.array(data["observation"]["head_camera"]["rgb"]), out_path=video_path)
# with open(file_path, "wb") as f:
#     pickle.dump(data, f)

# with open(file_path, "rb") as file:
#     data = pickle.load(file)

# print("Updated actions in the first transition:", data[0]["actions"])