import numpy as np
import math
from perception.perception import OpenVocab

defalt_obj_list = ['cap', 'hook', 'table', 'chair', 'person', 'sofa', 'stool', 'microwave', 'sink', 'cup']

# Key frame detection need to be improved 
# We can also use some learning method for this
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9137631 
def key_frame_detection(data, uni_sample=False, sampling_rate=1):
  # thresholds for finding keyframes
  keyframe_thresholds = {
      'position': 3.0,
      'orientation': 7.0,
      'arm': 1.0,
      'camera': 5.0
  }

  keyframe_trackers = {
      'position': 0,
      'orientation': 0,
      'arm': 0,
      'camera': 0
  }

  latest_values = {
      'odom': None,
      'joints': None,
      'state': None
  }

  keyframes = []

  # find keyframes
  arm_joints = ['wrist_extension', 'joint_lift', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_finger_left', 'joint_gripper_finger_right']
  camera_joints = ['joint_head_pan', 'joint_head_tilt']
  for i in range(len(data)):
      # TODO: optical flow
      
      # TODO: scene graph

      # TODO: other features
      
      # process joint changes
      frame_joints = data['joint_state'][i]
      if latest_values['joints'] is not None:
          for joint in arm_joints:
              keyframe_trackers['arm'] += abs(frame_joints[joint]['position'] - latest_values['joints'][joint]['position'])
          for joint in camera_joints:
              keyframe_trackers['camera'] += abs(frame_joints[joint]['position'] - latest_values['joints'][joint]['position'])
      
      # process odom changes
      frame_odom = data['odom'][i]
      if latest_values['odom'] is not None:
          keyframe_trackers['position'] += math.sqrt((frame_odom['position'][0] - latest_values['odom']['position'][0])**2 + (frame_odom['position'][1] - latest_values['odom']['position'][1])**2)
          keyframe_trackers['orientation'] += abs(frame_odom['orientation'] - latest_values['odom']['orientation'])

      # process state changes
      frame_state = data['state'][i]
      if (latest_values['state'] is not None and frame_state != latest_values['state']) or \
              keyframe_trackers['position'] > keyframe_thresholds['position'] or \
              keyframe_trackers['orientation'] > keyframe_thresholds['orientation'] or \
              keyframe_trackers['arm'] > keyframe_thresholds['arm'] or \
              keyframe_trackers['camera'] > keyframe_thresholds['camera']:
          keyframes.append(i)
          keyframe_trackers = {k: 0 for k in keyframe_trackers}

      latest_values['odom'] = frame_odom
      latest_values['joints'] = frame_joints
      latest_values['state'] = frame_state

  return keyframes

def generate_scene_graph(key_frame, vocab_list=defalt_obj_list, confidence=0.7):
    """
    Generates a scene graph from a key frame using an open vocabulary classifier.

    Args:
    key_frame (dict): Dictionary containing 'rgb', 'depth', and 'camera_info' for the key frame.
    vocab_list (list): List of vocabulary objects to recognize. Defaults to default_obj_list.
    confidence (float): Confidence threshold for the classifier. Defaults to 0.7.

    Returns:
    sg: The generated scene graph object.
    """
    # Create the open vocab classifier
    clf = OpenVocab(vocab_list, confidence=0.7)
    # Get scene graph from rgb, depth, and camera info
    sg = clf.scene_graph(key_frame['rgb'], key_frame['depth'], key_frame['camera_info'])
    return sg
    
def quaternion_to_euler_angles(x, y, z, w):
    """
    Converts quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).

    Args:
    x (float): The x component of the quaternion.
    y (float): The y component of the quaternion.
    z (float): The z component of the quaternion.
    w (float): The w component of the quaternion.

    Returns:
    tuple: A tuple containing the Euler angles (X, Y, Z) corresponding to roll, pitch, and yaw.
    """
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)

    t2 = np.where(t2<-1.0, -1.0, t2)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z 


