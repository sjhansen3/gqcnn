# coding: utf-8
import argparse
import logging
import IPython
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from math import sqrt, atan2, ceil

from autolab_core import RigidTransform, YamlConfig, Point

from gqcnn import RgbdImageState, ParallelJawGrasp
from gqcnn import Grasp2D
from gqcnn import CrossEntropyAntipodalGraspingPolicy, AntipodalDepthImageGraspSampler
from gqcnn import Visualizer as vis
from gqcnn import ImageMode, TrainingMode, PreprocMode, InputDataMode, GeneralConstants, ImageFileTemplates

from perception import ColorImage, CameraIntrinsics

FIGSIZE=8
class ProcessCornellData:
    def __init__(self, cfg):
        self._cfg = cfg

    def save_tensor(self, idx, color_image_tensor, pose_tensor, label_tensor):
        """ save a tensor to the dataset directory
        Params
        ---
        idx: the index of the image as labeled in the cornell directory (starts at 100)
        color_image_tensor: a numpy array ()
        pose_tensor: a numpy array representing the depth of each labeled gras (1,N)

        """
        idx-=100
        directory = self._cfg["dataset_dir"]
        
        color_filename = os.path.join(directory, ImageFileTemplates.color_im_tf_tensor_template+str(idx))
        pose_filename = os.path.join(directory, ImageFileTemplates.hand_poses_template+str(idx))
        label_filename = os.path.join(directory, self._cfg["target_metric_name"]+str(idx))
        
        print "saving color image to file: "
        print color_filename
        print pose_filename
        print label_filename
        
        np.savez_compressed(color_filename, color_image_tensor)
        np.savez_compressed(pose_filename, pose_tensor)
        np.savez_compressed(label_filename, label_tensor)

    def load(self, idx, is_positive_example):
        """ class to load the data from the cornell dataset
        Params
        ---
        idx: the suffix index of the cornell grasping image (starts at 100)
        is_positive_example: whether or not the example is labeled as a successful grasp or not
        Returns
        ---
        grasps: A list of autolab_core Grasp2D objects
        img
        """
        directory = self._cfg['raw_data_dir']
        if idx<10:
            file_name = "pcd000{}r".format(idx)
        elif idx<100:
            file_name = "pcd00{}r".format(idx)
        else:
            file_name = "pcd0{}r".format(idx)

        if is_positive_example:
            last= 'cpos.txt'
        else: 
            last= 'cneg.txt'
        
        #read the image
        img= plt.imread(directory + file_name +'.png')
        img = img*255
        img = img.astype('uint8')
        
        #print "image_shape{}".format(img.shape)
        img = ColorImage(img, frame='camera')  
        with open(directory+file_name[:-1]+last) as f:
            text = f.read()
            
        A = np.fromstring(text, sep=' ')
        A = np.reshape(A,(-1,8))

        x2 = A[:,0]
        y2 = A[:,1]
        x1 = A[:,2]
        y1 = A[:,3]
        x = A[:,4]
        y = A[:,5]
        x3 = A[:,6]
        y3 = A[:,7]

        vis.imshow(img)
        grasps = []
        intr = CameraIntrinsics('camera', fx=1, fy=1, cx=0, cy=0, width=640, height=480)
        for i in range(len(y3)-1):        
            #calculate the center point of the line
            center = np.array([(x[i]+x3[i])/2,(y[i]+y3[i])/2])
            grip_width_px = sqrt((x[i]-x3[i])**2+(y[i]-y3[i])**2)
            angle = atan2((y3[i]-y[i]),(x3[i]-x[i]))
    
            center_point = Point(center, frame='camera')
            grasp = Grasp2D(center_point,angle,depth=1,width=grip_width_px,camera_intr=intr)

            grasps.append(grasp)
        return grasps, img

    def view(self, image_tensor, pose_tensor):
        """ view the image and pose tensor
        """
        k = len(grasps)
        d = int(ceil(sqrt((k))))

        # display grasp transformed images
        vis.figure(size=(FIGSIZE,FIGSIZE))
        for i, image_tf in enumerate(image_tensor[:k,...]):
            depth = 1
            vis.subplot(d,d,i+1)
            vis.imshow(ColorImage(image_tf))
            vis.title('Image %d: d=%.3f' %(i, depth))

        # display grasp transformed images
        vis.figure(size=(FIGSIZE,FIGSIZE))
        for i in range(1,k):
            image_tf = image_tensor[i,...]
            depth = 1
            grasp = grasps[i]

            vis.subplot(d,2,2*i+1)
            vis.imshow(img)
            vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
            vis.title('Grasp %d: d=%.3f' %(i, depth))
            
            vis.subplot(d,2,2*i+2)
            vis.imshow(ColorImage(image_tf))
            vis.title('TF image %d: d=%.3f' %(i, depth))

    def view_hand_labels(self, A, img):
        """ Plot the image and hand labels
        Params
        ---
        A: The loaded matrix of labels from the text file
        img: The imag where the labels came from 
        """
        x2 = A[:,0]
        y2 = A[:,1]
        x1 = A[:,2]
        y1 = A[:,3]
        x = A[:,4]
        y = A[:,5]
        x3 = A[:,6]
        y3 = A[:,7]

        plt.figure()
        plt.imshow(img)
        for i in range(len(y3)-1):
            #show the box that was labeled by a human
            #blue indicates the locations of the gripper
            plt.plot([x1[i], x2[i]], [y1[i],y2[i]], 'r')
            plt.plot([x2[i], x3[i]], [y2[i],y3[i]], 'b')
            plt.plot([x[i], x3[i]], [y[i], y3[i]], 'r')
            plt.plot([x[i], x1[i]], [y[i], y1[i]], 'b')

    def grasps_to_tensors(self, grasps, image):
        """ Convert an image and grasps to tensors for training
        Params
        ---
        grasps: a list of Grasp2D objects
        image: the image which the tensors will be cropped out of
        Returns
        ---
        image_tensor: A (N,244,244,3) array of crops fro the image. Each grasp is encoded in the grasp
        pose_tensor: An (N,1) array of depths paired with each image
        """
        # parse params
        gqcnn_im_height = self._cfg['gqcnn_config']['im_height']
        gqcnn_im_width = self._cfg['gqcnn_config']['im_width']
        gqcnn_num_channels = self._cfg['gqcnn_config']['im_channels']
        gqcnn_pose_dim = 1 #normally set automatically in _read_data_params or _parse_config
        input_data_mode = self._cfg['input_data_mode']
        num_grasps = len(grasps)
        
        # allocate tensors
        image_tensor = np.zeros([num_grasps, gqcnn_im_height, gqcnn_im_width, gqcnn_num_channels],dtype='uint8')
        pose_tensor = np.zeros([num_grasps, gqcnn_pose_dim])

        for i, grasp in enumerate(grasps):
            scale = float(gqcnn_im_height) / (2*grasp.width)
            im_scaled = image.resize(scale)
            translation = scale * np.array([image.center[0] - grasp.center.data[1],
                                            image.center[1] - grasp.center.data[0]])
            im_tf = im_scaled.transform(translation, grasp.angle)
            im_tf = im_tf.crop(gqcnn_im_height, gqcnn_im_width)

            image_tensor[i,...] = im_tf.raw_data

            if input_data_mode == InputDataMode.TF_IMAGE:
                pose_tensor[i] = grasp.depth
            elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.center.x, grasp.center.y])
            else:
                raise ValueError('Input data mode %s not supported' %(input_data_mode))
        return image_tensor, pose_tensor

    def load_and_view(self, idx, is_positive_example):
        """ load, convert and view the pose tensor
        Params
        ---
        idx: the index of the image
        is_positive_example: whether or not the grasp is labeled as success or failure
        """
        grasps, img = self.load(idx, is_positive_example)

        plt.figure()
        image_tensor, pose_tensor = self.grasps_to_tensors(grasps, img)
        self.view(image_tensor, pose_tensor)

    def load_tensor(self, idx):
        """ load a set of image crops which have already been preprocessed and saved
        params
        ---
        idx: the index of the image as labeled by the cornell dataset
        Returns
        ---
        color_image_tensor: the color of the image
        pose_tensor: The depth with which to grab the object
        label_tensor: whether or not the labels are positive or negative
        """
        idx-=100
        directory = self._cfg["dataset_dir"]
        
        color_filename = os.path.join(directory, ImageFileTemplates.color_im_tf_tensor_template+str(idx)+".npz")
        pose_filename = os.path.join(directory, ImageFileTemplates.hand_poses_template+str(idx)+".npz")
        label_filename = os.path.join(directory, self._cfg["target_metric_name"]+str(idx)+".npz")
        
        print color_filename+".npz"
        print pose_filename+".npz"
        print label_filename+".npz"
        
        color_image_tensor = np.load(color_filename)['arr_0']
        pose_tensor = np.load(pose_filename)['arr_0']
        label_tensor = np.load(label_filename)['arr_0']
        
        return color_image_tensor, pose_tensor, label_tensor

if __name__=="__main__":

    # parse args
    parser = argparse.ArgumentParser(description='Load human grasp labels from the cornell dataset')
    parser.add_argument('--action', type=str, default='load', help='Which action is desired on the cornell dataset (load/save)')
    parser.add_argument('--config_filename', type=str, default='/Users/stephenhansen/Code/gqcnn/cfg/tools/color_training.yaml', help='path to the configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename

    action = args.action.lower()
    if action != "load" and action != "save":
        raise argparse.ArgumentTypeError("action must be either load or save")

    #configuration file for loading dataset directory
    cfg = YamlConfig(config_filename)

    if action=='save':
        #save the configuration for future debugging and use
        stored_config_path = os.path.join(cfg["dataset_dir"],"configuration_settings")
        cfg.save(stored_config_path)

        data_processor = ProcessCornellData(cfg)

        #generate and save the tensors
        for idx in range(102,104):
            grasps, img = data_processor.load(idx, is_positive_example=True)
            image_tensor, pose_tensor = data_processor.grasps_to_tensors(grasps, img)
            label_tensor_positive = np.zeros([len(grasps)])
            data_processor.save_tensor(idx, image_tensor, pose_tensor, label_tensor_positive)
    elif action =='load':
        # load the tensors
        idx = 109
        color_image_tensor, pose_tensor, label_tensor = load_tensor(cfg, idx)
        print color_image_tensor.shape
        length = len(color_image_tensor[:,0,0,0])
        d = int(ceil(sqrt(length)))
        for i in range(1,length):
            plt.subplot(d,d,i)
            print color_image_tensor[i,...].shape
            plt.imshow(color_image_tensor[i,...])