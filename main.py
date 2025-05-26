import numpy as np
from utils import key_frame_detection
from dataloader import read_bag, load_pickle
from llm_engine import NarrationEngine
import pickle

with open('config/constraints_summary.txt', 'r') as f:
    constraints = f.read()

if __name__ == '__main__':
    # bag_path = '/Users/zihanwang/Desktop/Research/Robot_Narration_LLM/reasoning/data/rosbag2_2024_05_15-01_10_46'
    # data = read_bag(bag_path, sampling_rate=1, scale_factor=0.5)

    for key in data.keys():
        print(key, len(data[key]))

    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f)

#     data = load_pickle('data.pkl')

    print('hello world')

    # key_frames = key_frame_detection(data)
    # print(len(key_frames))

    prior_used = 2
    narr_engine = NarrationEngine()
    trajectory_name = 'traj_1'
    narr_engine.add_trajectory(trajectory_name)
    for i in range(len(data['rgb'])):
        keyframe = {'rgb': data['rgb'][i], 'depth': data['depth'][i], 'odom': data['odom'][i], 'joint_state': data['joint_state'][i], 'camera_info': data['camera_info']}
        env_rgbd_summary = narr_engine.summarize_env_rgbd(keyframe)
        internal_summary = narr_engine.summarize_internal(keyframe)
        # planning_summary = narr_engine.summarize_planning(keyframe)
        summary_history = ''
        for timestamp, narr_hist_frame in narr_engine.narration_history[trajectory_name][i-prior_used:]:
            summary_history += 'Timestamp: ' + str(timestamp) + '\n' + narr_hist_frame + '\n'
        # summary_history = narr_engine.narration_history[trajectory_name][i-prior_used:]
        frame_summary = narr_engine.summarize_frame(env_summary=env_rgbd_summary, internal_summary=internal_summary, constrains=constraints, summary_history=summary_history)
        # print('#################env summary#######################')
        # print(env_rgbd_summary)
        # print('#################internal summary#######################')
        # print(internal_summary)
        # print('#################frame summary#######################')
        print(frame_summary)
        narr_engine.add_frame_narration(trajectory_name=trajectory_name, timestamp=i, narration=frame_summary)
        print('\n')
        if i == 3:
            break
        # narr_engine.summarize_env_rgbd()
        # narr_engine.summarize_env_rgbd()
        # narr_engine.summarize_planning()