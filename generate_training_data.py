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
import autolab_core.utils as utils

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

    def preprocess_grasp(self, filename, is_positive_example=True):
        """ preprocess a given image from the cornell dataset
        """
        grasps, img = self._load_cornell_data(filename, is_positive_example)
        if len(grasps) == 0:
            logging.error("No valid grasps found for image {}".format(filename))
            return False, False, False

        image_tensor, pose_tensor = self._grasps_to_tensors(grasps, img)
        
        if is_positive_example:
            label_tensor = np.ones([len(grasps)])
        else:
            label_tensor = np.zeros([len(grasps)])

        if self._cfg['vis_preprocess']:
            k = len(grasps)
            if k > self._cfg['vis_max']:
                k = self._cfg['vis_max']
            d = utils.sqrt_ceil(k)

            # display grasp transformed images
            vis.figure(size=(FIGSIZE,FIGSIZE))
            for i in range(k):
                image_tf = image_tensor[i,...]
                grasp = grasps[i]
                depth = pose_tensor[i]
                label = label_tensor[i]

                vis.subplot(k,2,2*i+1)
                vis.imshow(img)
                vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
                vis.title('Grasp %d: d=%.3f, label=%s' %(i, depth, label))
                
                vis.subplot(k,2,2*i+2)
                vis.imshow(ColorImage(image_tf))
                vis.title('TF image %d: d=%.3f, label=%s' %(i, depth, label))

        vis.show()
        return image_tensor, pose_tensor, label_tensor 

    def load_preprocessed_grasps(self, ident):
        """ load a set of image crops which have already been preprocessed and saved
        params
        ---
        idx: the index of the image as labeled by the cornell dataset
        Returns
        ---
        color_image_tensor: the color of the image
        pose_tensor: The depth with which to grab the object (N,1)
        label_tensor: whether or not the labels are positive or negative
        """
        #TODO get list of names, dont depend on them being in order or formatted
        directory = self._cfg["dataset_dir"]
        
        color_filename = os.path.join(directory, ImageFileTemplates.color_im_tf_tensor_template+ident+".npz")
        pose_filename = os.path.join(directory, ImageFileTemplates.hand_poses_template+ident+".npz")
        label_filename = os.path.join(directory, self._cfg["target_metric_name"]+ident+".npz")
        
        color_image_tensor = np.load(color_filename)['arr_0']
        pose_tensor = np.load(pose_filename)['arr_0']
        label_tensor = np.load(label_filename)['arr_0']
        
        return color_image_tensor, pose_tensor, label_tensor

    def save_tensor(self, ident, color_image_tensor, pose_tensor, label_tensor):
        """ save a tensor to the dataset directory
        Params
        ---
        idx: the index of the image as labeled in the cornell directory (starts at 100)
        color_image_tensor: a stack of numpy array's for each image
        pose_tensor: a numpy array representing the depth of each labeled gras (1,N)
        """
        directory = self._cfg["dataset_dir"]
        
        color_filename = os.path.join(directory, ImageFileTemplates.color_im_tf_tensor_template+ident)
        pose_filename = os.path.join(directory, ImageFileTemplates.hand_poses_template+ident)
        label_filename = os.path.join(directory, self._cfg["target_metric_name"]+ident)
        
        logging.debug("saving color image to file: ")
        logging.debug(color_filename)
        logging.debug(pose_filename)
        logging.debug(label_filename)
        
        np.savez_compressed(color_filename, color_image_tensor)
        np.savez_compressed(pose_filename, pose_tensor)
        np.savez_compressed(label_filename, label_tensor)

    def _load_cornell_data(self, file_name, is_positive_example):
        """ class to load a set of grasps associated with a single image from the cornell dataset
        Params
        ---
        file_name: the name of the image file
        is_positive_example: whether or not the example is labeled as a successful grasp or not
        Returns
        ---
        grasps: A list of autolab_core Grasp2D objects
        img: the full image from the cornell dataset
        """
        directory = self._cfg['raw_data_dir']
        if is_positive_example:
            last= 'cpos.txt'
        else: 
            last= 'cneg.txt'
        
        #read the image
        logging.debug("directory name {}".format(directory + file_name))
        try:
            img= plt.imread(directory + file_name)

        except IOError as e:
            logging.warn(e)
            return [], []

        img = img*255
        img = img.astype('uint8')
        if img.shape[2] > 3:
            img = img[:,:,0:3]
        

        img = ColorImage(img, frame='camera')  
        with open(directory+file_name[:-5]+last) as f:
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

        if self._cfg['vis_preprocess']:
            vis.figure()
            vis.imshow(img)

        grasps = []

        #set the camera intrinsics so the grasp passes through as pixels. Kinda hacky
        intr = CameraIntrinsics('camera', fx=1, fy=1, cx=0, cy=0, width=640, height=480)
        
        #loop through all boxes which a human labeled
        for i in range(len(y3)-1):        
            #calculate the center point of the line
            center_1 = np.array([(x[i]+x3[i])/2,(y[i]+y3[i])/2])
            center_2 = np.array([(x1[i]+x2[i])/2,(y1[i]+y2[i])/2])

            grip_width_px = sqrt((x[i]-x3[i])**2+(y[i]-y3[i])**2)
            angle = atan2((y3[i]-y[i]),(x3[i]-x[i]))
            
            if is_infinite(grip_width_px) or is_infinite(center_1) or is_infinite(center_2) or is_infinite(angle):
                logging.warn("Skipping grasp {} of picture {} for nan or inf value".format(i, file_name))
                logging.warn("x[i], {} x3[i], {} y[i], {} y3[i], {}".format(x[i], x3[i], y[i], y3[i]))
                continue

            num_samples = self._cfg['num_samples']
            multiples = np.arange(0,num_samples)/float(num_samples)
            for i_sample in range(num_samples):
                center = center_1+(center_2-center_1)*multiples[i_sample]
                center_point = Point(center, frame='camera')
                grasp = Grasp2D(center_point,angle,depth=1,width=grip_width_px,camera_intr=intr)
                grasps.append(grasp)
                if self._cfg['vis_preprocess']:
                    vis.grasp(grasp)
                    
                    #show the human labeled box
                    vis.plot([x1[i], x2[i]], [y1[i],y2[i]], 'r')
                    vis.plot([x2[i], x3[i]], [y2[i],y3[i]], 'b')
                    vis.plot([x[i], x3[i]], [y[i], y3[i]], 'r')
                    vis.plot([x[i], x1[i]], [y[i], y1[i]], 'b')

        return grasps, img

    def _view_generated_tensors(self, image_tensor, pose_tensor, label_tensor):
        """ view the image and pose tensor
        """
        k = len(pose_tensor[0])
        if k > self._cfg['vis_max']:
            k = self._cfg['vis_max']
        d = utils.sqrt_ceil(k)

        # display grasp transformed images
        vis.figure(size=(FIGSIZE,FIGSIZE))
        for i in range(k):
            image_tf = image_tensor[i,...]
            depth = pose_tensor[i]
            label = label_tensor[i]
            
            vis.subplot(d,d,i)
            vis.imshow(ColorImage(image_tf))
            vis.title('TF image %d: d=%.3f, label=%s' %(i, depth, label))

        vis.show()

    def _grasps_to_tensors(self, grasps, image):
        """ Convert a set of grasps and single image to image and pose tensors for training
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
            scale = float(gqcnn_im_height) / (2*grasp.width+1e-6)
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

def is_infinite(num):
    return np.any(np.isinf(num)) or np.any(np.isnan(num))

if __name__=="__main__":
    # set up logger
    logging.getLogger().setLevel(logging.INFO)

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
    data_processor = ProcessCornellData(cfg)

    if action=='save':
        #save the configuration for future debugging and use
        stored_config_path = os.path.join(cfg["dataset_dir"],"configuration_settings")
        cfg.save(stored_config_path)

        #process the cornell data and save the tensors to file for training
        image_tensors = []#np.zeros((1000,64,64,3))
        pose_tensors = []#np.zeros((1000,7))
        label_tensors = []#np.zeros((1000,))

        all_filenames = os.listdir(cfg['raw_data_dir'])

        im_filenames = [f for f in all_filenames if f.find(".png") > -1]
        im_filenames = im_filenames
        for idx, filename in enumerate(im_filenames):
            img_tensor1, pose_tensor1, label_tensor1 = data_processor.preprocess_grasp(filename, is_positive_example=False)
            if np.all(img_tensor1):
                print "shape of img_tensor1 {}".format(img_tensor1.shape)
                # print "shape of pose_tensor1 {}".format(pose_tensor1.shape)
                # print "shape of label_tensor1 {}".format(label_tensor1.shape)
                image_tensors.append(img_tensor1)
                pose_tensors.append(pose_tensor1)
                label_tensors.append(label_tensor1)

            img_tensor2, pose_tensor2, label_tensor2 = data_processor.preprocess_grasp(filename, is_positive_example=True)
            if np.all(img_tensor2):
                print "shape of img_tensor2 {}".format(img_tensor2.shape)
                # print "shape of pose_tensor2 {}".format(pose_tensor2.shape)
                # print "shape of label_tensor2 {}".format(label_tensor2.shape)

                image_tensors.append(img_tensor2)
                pose_tensors.append(pose_tensor2)
                label_tensors.append(label_tensor2)

        # for tensor in label_tensors:
        #     print "tensor shape {}".format(tensor.shape)

        # import pdb; pdb.set_trace()
        image_tensors = np.vstack(image_tensors)
        pose_tensors = np.vstack(pose_tensors)
        pose_tensors = np.hstack([np.zeros((len(pose_tensors), 2)), pose_tensors, np.zeros((len(pose_tensors), 4))])
        label_tensors = np.hstack(label_tensors)

        print "shape of the final image tensor {}".format(image_tensors.shape)
        print "shape of the pose tensor {}".format(pose_tensors.shape)
        print "label tensor final shape {}".format(label_tensors.shape)

        N = int(ceil(len(image_tensors)/1000.0))
        image_tensors = np.array_split(image_tensors, N)
        pose_tensors = np.array_split(pose_tensors, N)
        label_tensors = np.array_split(label_tensors, N)

        for i, (img_tensor, pose_tensor, lbl_tensor) in enumerate(zip(image_tensors, pose_tensors, label_tensors)):
            ident = str(i).zfill(4)
            print "identity {}".format(ident)
            print "shape of img_tensor {}".format(img_tensor.shape)
            print "shape of pose_tensor {}".format(pose_tensor.shape)
            print "shape of lbl_tensor {}".format(lbl_tensor.shape)

            data_processor.save_tensor(ident, img_tensor, pose_tensor, lbl_tensor)

    
    elif action =='load':
        #load the processed and saved tensors which have been saved for training
        color_image_tensor, pose_tensor, label_tensor = data_processor.load_preprocessed_grasps(idx)
        print "the shape of the loaded color_image tensor: {}".format(color_image_tensor.shape)
        length = len(color_image_tensor[:,0,0,0])
        d = int(ceil(sqrt(length)))
        for i in range(1,length):
            plt.subplot(d,d,i)
            print color_image_tensor[i,...].shape
            plt.imshow(color_image_tensor[i,...])
        plt.show()