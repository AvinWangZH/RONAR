from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics import YOLOWorld
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class OpenVocab:
    def __init__(self, classes, confidence=0.75):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = classes
        self.detector = YOLOWorld('yolov8l-world.pt')
        self.detector.set_classes(classes)
        self.segmenter = FastSAM('FastSAM-x.pt')
        self.confidence = confidence

    def scene_graph(self, image, depth, camera, depth_mask=None):
        # define nodes for scene graph
        class Node():
            def __init__(self, x, y, z, label):
                self.x = x
                self.y = y
                self.z = z
                self.label = label

            def __repr__(self):
                return f'{self.label}: ({self.x}, {self.y}, {self.z})'

        # calculate size to fit ultralytics model
        original_size = image.shape[:2]
        modified_size = (original_size[0] - original_size[0] % 32, original_size[1] - original_size[1] % 32)
        height, width = modified_size
        cropped_image = image[:modified_size[0], :modified_size[1], :]

        # predict objects
        results = self.detector.predict(cropped_image, imgsz=modified_size, conf=self.confidence)
        frame_classes = results[0].boxes.cls.tolist()
        frame_boxes = results[0].boxes.xyxy.tolist()

        nodes = []
        nodes.append(Node(0, 0, 0, 'robot_camera'))
        instances = {}
        # segment
        for detection in range(len(frame_boxes)):
            everything_results = self.segmenter(cropped_image, imgsz=modified_size, device=self.device)
            prompt_process = FastSAMPrompt(cropped_image, everything_results, device=self.device)
            ann = prompt_process.box_prompt(bbox=frame_boxes[detection])
            segment = ann[0].masks.data[0]
            segment_mask = ann[0].masks.data[0].bool()

            # get object center
            z = depth[:modified_size[0], :modified_size[1]]
            x = (np.tile(range(width), height).reshape(height, width) - camera['cx']) * z / camera['fx']
            y = (camera['cy'] - np.repeat(range(height), width).reshape(height, width)) * z / camera['fy']
            
            object_x = torch.mean(torch.masked_select(torch.from_numpy(x.copy()), segment_mask).float())
            object_y = torch.mean(torch.masked_select(torch.from_numpy(y.copy()), segment_mask).float())
            object_z = torch.mean(torch.masked_select(torch.from_numpy(np.array(z, copy=True, dtype=np.uint8)), segment_mask).float())

            if depth_mask is not None and object_z.float() > depth_mask:
                continue

            label = self.classes[int(frame_classes[detection])]

            if label in instances:
                instances[label] += 1
            else:
                instances[label] = 0

            # print('{}, ({}, {}, {})'.format(label, object_x.item(), object_y.item(), object_z.item()))

            object_node = Node(int(torch.round(object_x).item()), int(torch.round(object_y).item()), int(torch.round(object_z).item()), f'{label}_{instances[label]}')
            nodes.append(object_node)

        pos = []
        for i in range(len(nodes)):
            local_pos = []
            for j in range(i, len(nodes)):
                if i != j:
                    diffs = [abs((nodes[i].x - nodes[j].x)), abs((nodes[i].y - nodes[j].y)), abs((nodes[i].z - nodes[j].z))]
                    dom_index = max(range(len(diffs)), key=diffs.__getitem__)
                    dom = ''
                    if nodes[i].x > nodes[j].x:
                        if dom_index == 0:
                            dom = 'right of'
                    else:
                        if dom_index == 0:
                            dom = 'left of'
                    if nodes[i].y > nodes[j].y:
                        if dom_index == 1:
                            dom = 'above'
                    else:
                        if dom_index == 1:
                            dom = 'below'
                    if nodes[i].z > nodes[j].z:
                        if dom_index == 2:
                            dom = 'behind'
                    else:
                        if dom_index == 2:
                            dom = 'in front of'
                    local_pos.append((nodes[j].label, dom))
            for k in range(len(local_pos)):
                pos.append((nodes[i].label, local_pos[k][0], local_pos[k][1]))

        return {
            'nodes': [n.label for n in nodes],
            'edges': pos,
            'distance_from_camera': [(n.label, f'{n.z / 1000} meters') for n in nodes]
        }
    

    def scene_graph_3d(self, image, depth, camera, depth_mask=None, visualize=False):
        """
        Creates a scene graph from an image

        Args:
            image (numpy array): image to be processed
            depth (numpy array): depth map of the image
            camera (dict): dict containing cx (center x), cy (center y), fx (focal length x), fy (focal length y)
            depth_mask (float): maximum depth for objects to be considered
            visualize (bool): whether to visualize the scene graph

        Returns:
            pos (list): list of positional relationships between objects
        """
        # define nodes for scene graph
        class Node():
            def __init__(self, x, y, z, label):
                self.x = x
                self.y = y
                self.z = z
                self.label = label

            def __repr__(self):
                return f'{self.label}: ({self.x}, {self.y}, {self.z})'

        # calculate size to fit ultralytics model
        original_size = image.shape[:2]
        modified_size = (original_size[0] - original_size[0] % 32, original_size[1] - original_size[1] % 32)
        height, width = modified_size
        cropped_image = image[:modified_size[0], :modified_size[1], :]

        # predict objects
        results = self.detector.predict(cropped_image, imgsz=modified_size, conf=self.confidence)
        frame_classes = results[0].boxes.cls.tolist()
        frame_boxes = results[0].boxes.xyxy.tolist()

        # visualize bounding boxes
        if visualize:
            fig, ax = plt.subplots(1)
            ax.imshow(cropped_image)

            for i in range(len(frame_classes)):
                x1, y1, x2, y2 = frame_boxes[i]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            plt.show()


        nodes = []
        # segment
        for detection in range(len(frame_boxes)):
            everything_results = self.segmenter(cropped_image, imgsz=modified_size, device=self.device)
            prompt_process = FastSAMPrompt(cropped_image, everything_results, device=self.device)
            ann = prompt_process.box_prompt(bbox=frame_boxes[detection])
            segment = ann[0].masks.data[0]
            segment_mask = ann[0].masks.data[0].bool()

            # visualize segmentation
            if visualize:
                color = list(np.random.choice(range(256), size=3))
                color_matrix = np.zeros((modified_size[0], modified_size[1], 3), dtype=np.uint8)
                color_original = results[0].orig_img.copy()
                color_matrix[segment_mask] = color
                color_original[segment_mask] = color
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(color_matrix)
                ax2.imshow(color_original)
                plt.show()

            # get object center
            z = depth[:modified_size[0], :modified_size[1]]
            x = (np.tile(range(width), height).reshape(height, width) - camera['cx']) * z / camera['fx']
            y = (camera['cy'] - np.repeat(range(height), width).reshape(height, width)) * z / camera['fy']
            
            object_x = torch.mean(torch.masked_select(torch.from_numpy(x.copy()), segment_mask).float())
            object_y = torch.mean(torch.masked_select(torch.from_numpy(y.copy()), segment_mask).float())
            object_z = torch.mean(torch.masked_select(torch.from_numpy(z.copy()), segment_mask).float())

            if depth_mask is not None and object_z.float() > depth_mask:
                continue

            label = self.classes[int(frame_classes[detection])]

            print('{}, ({}, {}, {})'.format(label, object_x.item(), object_y.item(), object_z.item()))

            object_node = Node(torch.round(object_x).int(), torch.round(object_y).int(), torch.round(object_z).int(), self.classes[int(frame_classes[detection])])
            nodes.append(object_node)

        # create scene graph
        if len(nodes) < 2:
            return []

        pos = []
        for i in range(len(nodes)):
            local_pos = []
            for j in range(len(nodes)):
                if i != j:
                    diffs = [abs((nodes[i].x - nodes[j].x).item()), abs((nodes[i].y - nodes[j].y).item()), abs((nodes[i].z - nodes[j].z).item())]
                    dom_index = max(range(len(diffs)), key=diffs.__getitem__)
                    dom = ''
                    if nodes[i].x > nodes[j].x:
                        if dom_index == 0:
                            dom = 'right of'
                    else:
                        if dom_index == 0:
                            dom = 'left of'
                    if nodes[i].y > nodes[j].y:
                        if dom_index == 1:
                            dom = 'above'
                    else:
                        if dom_index == 1:
                            dom = 'below'
                    if nodes[i].z > nodes[j].z:
                        if dom_index == 2:
                            dom = 'behind'
                    else:
                        if dom_index == 2:
                            dom = 'in front of'
                    local_pos.append('{} the {}'.format(dom, nodes[j].label))
            local_text = 'The {} is '.format(nodes[i].label)
            for k in range(len(local_pos)):
                if k == 0:
                    local_text += local_pos[k]
                elif k == len(local_pos) - 1 and len(local_pos) <= 2:
                    local_text += ' and {}.'.format(local_pos[k])
                elif k == len(local_pos) - 1:
                    local_text += ', and {}.'.format(local_pos[k])
                else:
                    local_text += ', {}'.format(local_pos[k])
            pos.append(local_text)

        return pos