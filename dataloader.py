import math
import cv2
import utils
import numpy as np
import pickle
import json
from datetime import datetime
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# TODO: transform all ros bags to hdf5

def load_pickle(file_pth):
  with open(file_pth, 'rb') as f:
    data = pickle.load(f)
  return data

def read_txt(file_pth):
  with open(file_pth, 'r') as f:
    data = f.read()
  return data

def load_json(file_pth):
  with open(file_pth, 'r') as f:
    data = json.load(f)
  return data
    
def read_bag(bag_path, sampling_rate=1, data_config=None, scale_factor=1):
    """
    Reads data from a ROS2 bag file with specified sampling rate.

    Args:
    bag_path (str): Path to the ROS2 bag file.
    sampling_rate (int): Frames per second to sample. Defaults to 1.
    data_config (dict): Dictionary to configure which data types to read. Defaults to None.

    Returns:
    dict: A dictionary containing sampled data.
    """

    # Load custom interfaces
    add_types = {}
    add_types.update(get_types_from_msg('string[] string_list', 'stretch_narration_interfaces/msg/StringList'))
    add_types.update(get_types_from_msg('StringList[] string_list_list', 'stretch_narration_interfaces/msg/StringListList'))
    
    # Register custom types
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(add_types)

    data = {
        'camera_info': None,
        'rgb': [], 
        'depth': [],
        'odom': [],
        'joint_state': [],
        'state': [],
        'state_history': []
    }

    # get the first frame from each second
    if data_config == None:
        data_config = {
            'rgb': True,
            'depth': True,
            'odom': True,
            'joint_state': True,
            # 'state': True,
            # 'state_history': True
        }

    n_frames = 0

    with Reader(bag_path) as reader:
        last_sampled_time = None
        for connection, timestamp, rawdata in reader.messages():
            if sampling_rate == -1:
                if connection.topic == '/camera/color/image_raw' and data_config['rgb']:
                        # preprocess - reshape and rotate
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        image = cv2.rotate(msg.data.reshape(msg.height, msg.width, 3), cv2.ROTATE_90_CLOCKWISE)
                        image_scaled = cv2.resize(image, (int(msg.height*scale_factor), int(msg.width*scale_factor)))
                        data['rgb'].append(image_scaled)
                if connection.topic == '/camera/aligned_depth_to_color/image_raw' and data_config['depth']:
                        # preprocess - reshape and rotate
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        depth_data = np.frombuffer(msg.data, dtype=np.uint16)
                        depth_image = cv2.rotate(np.reshape(depth_data.astype(np.float64), (msg.height, msg.width)), cv2.ROTATE_90_CLOCKWISE)
                        depth_image_scaled = cv2.resize(depth_image, (int(msg.height*scale_factor), int(msg.width*scale_factor)))
                        data['depth'].append(depth_image_scaled)
                if connection.topic == '/camera/aligned_depth_to_color/camera_info' and data['camera_info'] is None:
                        # preprocess - get intrinsics and rotate
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        k = msg.k
                        camera_info = {
                            'fx': k[0],
                            'cx': k[2],
                            'fy': k[4],
                            'cy': k[5]
                        }
                        camera_info_rotated = {
                            'cx': camera_info['cy'],
                            'cy': camera_info['cx'],
                            'fx': camera_info['fy'],
                            'fy': camera_info['fx']
                        }
                        data['camera_info'] = camera_info_rotated
                if connection.topic == '/current_state' and data_config['state']:
                        state = typestore.deserialize_cdr(rawdata, connection.msgtype).data
                        data['state'].append(state)
                if connection.topic == '/state_history' and data_config['state_history']:
                        # preprocess - load into list of lists
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                        state_history = [msg.string_list_list[i].string_list for i in range(len(msg.string_list_list))]
                        data['state_history'].append(state_history)
                if connection.topic == '/stretch/joint_states' and data_config['joint_state']:
                        # preprocess - map joint to position, velocity, effort                
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        joints = {}
                        for i in range(len(msg.name)):
                            joints[msg.name[i]] = {
                                'position': msg.position[i],
                                'velocity': msg.velocity[i],
                                'effort': msg.effort[i]
                            }
                        data['joint_state'].append(joints)
                if connection.topic == '/odom' and data_config['odom']:
                        # preprocess
                        # keep (x, y) for position, orientation
                        # keep single velocity and angular velocity values
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                        quaternion = msg.pose.pose.orientation
                        orientation = utils.quaternion_to_euler_angles(quaternion.x, quaternion.y, quaternion.z, quaternion.w)[2]
                        velocity = msg.twist.twist.linear.x
                        angular_velocity = msg.twist.twist.angular.z

                        data['odom'].append({
                            'position': position,
                            'orientation': orientation,
                            'velocity': velocity,
                            'angular_velocity': angular_velocity
                        })
                
            else:
                if last_sampled_time == None:
                    last_sampled_time = timestamp # Set the initial sampled time

                # Check if enough time has passed since the last sample
                elif (timestamp - last_sampled_time) * 1e-9 >= 1/sampling_rate:
                    if len(data['rgb']) % 10 == 0:
                        print(len(data['rgb']))
                    if connection.topic == '/camera/color/image_raw' and data_config['rgb']:
                        # preprocess - reshape and rotate
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        image = cv2.rotate(msg.data.reshape(msg.height, msg.width, 3), cv2.ROTATE_90_CLOCKWISE)
                        image_scaled = cv2.resize(image, (int(msg.height*scale_factor), int(msg.width*scale_factor)))
                        data['rgb'].append(image_scaled)
                        data_config['rgb'] = False
                        
                    if connection.topic == '/camera/aligned_depth_to_color/image_raw' and data_config['depth']:
                        # preprocess - reshape and rotate
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        depth_data = np.frombuffer(msg.data, dtype=np.uint16)
                        depth_image = cv2.rotate(np.reshape(depth_data.astype(np.float64), (msg.height, msg.width)), cv2.ROTATE_90_CLOCKWISE)
                        depth_image_scaled = cv2.resize(depth_image, (int(msg.height*scale_factor), int(msg.width*scale_factor)))
                        data['depth'].append(depth_image_scaled)
                        data_config['depth'] = False

                    if connection.topic == '/camera/aligned_depth_to_color/camera_info' and data['camera_info'] is None:
                        # preprocess - get intrinsics and rotate
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        k = msg.k
                        camera_info = {
                            'fx': k[0],
                            'cx': k[2],
                            'fy': k[4],
                            'cy': k[5]
                        }
                        camera_info_rotated = {
                            'cx': camera_info['cy'],
                            'cy': camera_info['cx'],
                            'fx': camera_info['fy'],
                            'fy': camera_info['fx']
                        }
                        data['camera_info'] = camera_info_rotated

                    if connection.topic == '/current_state' and data_config['state']:
                        state = typestore.deserialize_cdr(rawdata, connection.msgtype).data
                        data['state'].append(state)
                        data_config['state'] = False

                    if connection.topic == '/state_history' and data_config['state_history']:
                        # preprocess - load into list of lists
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                        state_history = [msg.string_list_list[i].string_list for i in range(len(msg.string_list_list))]
                        data['state_history'].append(state_history)
                        data_config['state_history'] = False

                    if connection.topic == '/stretch/joint_states' and data_config['joint_state']:
                        # preprocess - map joint to position, velocity, effort                
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        joints = {}
                        for i in range(len(msg.name)):
                            joints[msg.name[i]] = {
                                'position': msg.position[i],
                                'velocity': msg.velocity[i],
                                'effort': msg.effort[i]
                            }

                        data['joint_state'].append(joints)
                        data_config['joint_state'] = False

                    if connection.topic == '/odom' and data_config['odom']:
                        # preprocess
                        # keep (x, y) for position, orientation
                        # keep single velocity and angular velocity values
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                        quaternion = msg.pose.pose.orientation
                        orientation = utils.quaternion_to_euler_angles(quaternion.x, quaternion.y, quaternion.z, quaternion.w)[2]
                        velocity = msg.twist.twist.linear.x
                        angular_velocity = msg.twist.twist.angular.z

                        data['odom'].append({
                            'position': position,
                            'orientation': orientation,
                            'velocity': velocity,
                            'angular_velocity': angular_velocity
                        })

                        data_config['odom'] = False
                    
                    # Reset the data configuration and update the last sampled time
                    if all([not v for v in data_config.values()]):
                        data_config = {k: True for k in data_config}
                        last_sampled_time = timestamp
                        n_frames += 1
    return data