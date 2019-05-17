#coding=utf-8
import numpy as np
import json

'''
#detectron import ...
from caffe2.python import workspace

from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import core.test_engine as infer_engine

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
'''

#tensorflow import ...
"""Functions to export object detection inference graph."""
from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
  

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class Model():
    
    def __init__(self,PATH_TO_FROZEN_GRAPH,PATH_TO_LABELS):
    
        self.score_thresh = 0.1
        self.class_nms_thresh = 0.85
        NUM_CLASSES = 1
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True 
        config.gpu_options.per_process_gpu_memory_fraction = 0.03
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            sess = tf.Session(config=config)
            
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            self.detection = lambda image : sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        
        #Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)  

        print ("model is ok")

    def predict(self,im):
        #print('type(im)')
        #print(type(im))
        #class_str_list = []
        data_list = []
        
        #image_np = self.load_image_into_numpy_array(im)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(im, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(im)
        #print(output_dict)
        '''
        with c2_utils.NamedCudaScope(self.gpu_id):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, im, None, None
            )
            
        #get box classes
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = self.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.score_thresh:
            return data_list
        #get score
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        '''
        
        num = 0
        for i in xrange(output_dict['detection_scores'].shape[0]):
            #print(output_dict['detection_scores'][i])
            if output_dict['detection_scores'][i] < 0.1: #self.score_thresh:
                num = i
                break;
        #print (num)

        scorse = output_dict['detection_scores'][0:num]
        #print('detection_scores')
        #print(output_dict['detection_scores'][0:num])
        boxes = output_dict['detection_boxes'][0:num,:]
        #print('detection_boxes')
        #print(output_dict['detection_boxes'][0:num])
        classes =  output_dict['detection_classes'][0:num]
        #print('detection_classes')
        #print(output_dict['detection_classes'][0:num])
        if boxes is None or boxes.shape[0] == 0 or max(scorse[:]) < self.score_thresh:
            return data_list
            
        #boxes  0~1 to width,height
        #ymin, xmin, ymax, xmax = box
        #im_width, im_height = im.size
        im_height ,im_width , _ = im.shape
        #(left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
        boxes[:, 0] = boxes[:, 0] * im_height  #ymin
        boxes[:, 1] = boxes[:, 1] * im_width   #xmin
        boxes[:, 2] = boxes[:, 2] * im_height  #ymax
        boxes[:, 3] = boxes[:, 3] * im_width   #xmax
        
        #get score
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        
        
        #print('scorse')
        #print(scorse)
        if (len(sorted_inds) > 0):        
            nmsIndex = self.nms_between_classes(boxes, scorse,self.class_nms_thresh)  #阈值为0.9，阈值越大，过滤的越少 
            #print(nmsIndex)
            for i in xrange(len(nmsIndex)):                
                bbox = boxes[nmsIndex[i], :4]
                #print('bbox')
                #print(bbox)
                score = scorse[nmsIndex[i]]
                if score < self.score_thresh:
                    continue
                #get class-str
                #class_str = self.get_class_string(classes[nmsIndex[i]], score, self.dummy_coco_dataset)
                if classes[i] in self.category_index.keys():
                    class_str = self.category_index[classes[i]]['name']
                else:
                    class_str = 'N/A'
                
                single_data = {"cls":class_str,"score":float('%.2f' % score),"bbox":{"xmin":int(bbox[1]),"ymin":int(bbox[0]),"xmax":int(bbox[3]),"ymax":int(bbox[2])}}
                #print(single_data)
                data_list.append(single_data)        
        
                '''cv2.rectangle(result2,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,255,0),1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                ((txt_w, txt_h), _) = cv2.getTextSize(class_str, font, 0.55, 1)            
                txt_tl = int(bbox[0]), int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(result2, class_str, txt_tl, font, 0.55, (218, 227, 218), lineType=cv2.LINE_AA)
                txt_tl = int(bbox[0])+txt_w, int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(result2, ('%.2f' % score), txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)'''
        #cv2.imwrite("test2.jpg", result2)
        
        #construcrion - data_list

        
        return data_list
        
    def convert_from_cls_format(self,cls_boxes, cls_segms, cls_keyps):
        """Convert from the class boxes/segms/keyps format generated by the testing
        code.
        """
        box_list = [b for b in cls_boxes if len(b) > 0]
        if len(box_list) > 0:
            boxes = np.concatenate(box_list)
        else:
            boxes = None
        if cls_segms is not None:
            segms = [s for slist in cls_segms for s in slist]
        else:
            segms = None
        if cls_keyps is not None:
            keyps = [k for klist in cls_keyps for k in klist]
        else:
            keyps = None
        classes = []
        for j in range(len(cls_boxes)):
            classes += [j] * len(cls_boxes[j])
        return boxes, segms, keyps, classes
        
    def get_class_string(self,class_index, score, dataset):
        class_text = dataset.classes[class_index] if dataset is not None else \
            'id{:d}'.format(class_index)
        #return class_text + ' {:0.2f}'.format(score).lstrip('0')
        return class_text
    def nms_between_classes(self,boxes, sorces, threshold):
        if boxes.size==0:
            return np.empty((0,3))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        #s = boxes[:,4]
        s = sorces
        area = (x2-x1+1) * (y2-y1+1)
        I = np.argsort(s)
        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while I.size>0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            idx = I[0:-1]
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])
            w = np.maximum(0.0, xx2-xx1+1)
            h = np.maximum(0.0, yy2-yy1+1)
            inter = w * h        
            o = inter / (area[i] + area[idx] - inter)
            I = I[np.where(o<=threshold)]
        pick = pick[0:counter]  #返回nms后的索引
        return pick
    #help code
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size  
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
          
    def run_inference_for_single_image(self,image):
        '''
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        '''
        # Run inference

        #output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        output_dict = self.detection(image)


        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
    
