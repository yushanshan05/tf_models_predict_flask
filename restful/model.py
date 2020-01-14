#coding=utf-8
import numpy as np
import json
import logging

import cv2 
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
model_logger = logging.getLogger('oilsteal.model')


class Model():
    
    def __init__(self,PATH_TO_FROZEN_GRAPH,PATH_TO_LABELS):
    
        self.score_thresh = 0.1
        self.class_nms_thresh = 0.85
        NUM_CLASSES = 1
        
       	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True 
        config.gpu_options.per_process_gpu_memory_fraction = 0.03
        
        detection_graph = tf.Graph()
        with tf.device('/gpu:0'):   
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
        
        self.crossAreaRatios = 0.03
        self.PolyDebug = False
        if self.PolyDebug == True:
            self.img=np.zeros((1080,1920,3))            

        print ("model is ok")

    def predict(self,im, x_position, y_position):
        
        position_num = 0         
        if len(x_position) > 2 :            
            position_num = len(x_position)
        
        if (self.PolyDebug == True and position_num > 2):
            ll=[]
            for i in range(position_num):
                ll.append([x_position[i],y_position[i]])
            cv2.fillConvexPoly(self.img, np.array(ll, np.int32), (255,255,0)) 
        
        
        data_list = []
        
        #image_np = self.load_image_into_numpy_array(im)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(im, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(im)
        #print(output_dict)       
        
        
        num = 0
        for i in xrange(output_dict['detection_scores'].shape[0]):
            #print(output_dict['detection_scores'][i])
            if output_dict['detection_scores'][i] < self.score_thresh:
                num = i
                break;
        #print (num)
        num = output_dict['detection_scores'].shape[0]
        scorse = output_dict['detection_scores'][0:num]       
        boxes = output_dict['detection_boxes'][0:num,:]        
        classes =  output_dict['detection_classes'][0:num]
        
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
        
        #print("#######nms before:",len(sorted_inds))
        if (len(sorted_inds) > 0):        
            nmsIndex = self.nms_between_classes(boxes, scorse,self.class_nms_thresh)  #阈值为0.9，阈值越大，过滤的越少 
            #print(nmsIndex)            
            
            for i in xrange(len(nmsIndex)):                
                bbox = boxes[nmsIndex[i], :4]                
                score = scorse[nmsIndex[i]]
                if score < self.score_thresh:
                    continue
                #get class-str
                #class_str = self.get_class_string(classes[nmsIndex[i]], score, self.dummy_coco_dataset)
                if classes[i] in self.category_index.keys():
                    class_str = self.category_index[classes[i]]['name']
                else:
                    class_str = 'N/A'
                
                if len(x_position) > 2 :
                    bbox_x = [int(bbox[1]), int(bbox[3]), int(bbox[3]), int(bbox[1])]
                    bbox_y = [int(bbox[0]), int(bbox[0]), int(bbox[2]), int(bbox[2])]
                    if self.IsFilterByElectronicFence(bbox_x, bbox_y, x_position, y_position):
                        continue
                    
                
                single_data = {"cls":class_str,"score":float('%.2f' % score),"bbox":{"xmin":int(bbox[1]),"ymin":int(bbox[0]),"xmax":int(bbox[3]),"ymax":int(bbox[2])}}
                #print(single_data)
                data_list.append(single_data)  
                
        #construcrion - data_list
        if self.PolyDebug == True:
            cv2.imwrite("1.jpg", self.img)
            
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
        
        
    def IsFilterByElectronicFence(self,bbox_x, bbox_y, x_position, y_position):
        
        if self.PolyDebug == True:
            self.drawInfo(bbox_x, bbox_y)
            
        model_logger.info("electornic: x:{}, y:{}".format(x_position,y_position))
            
        x_position_max_e = 0
        x_position_min_e = 0
        y_position_max_e = 0
        y_position_min_e = 0
        position_num_e = 0 
        
        if len(x_position) > 2 :
            x_position_max_e = max(x_position)
            x_position_min_e = min(x_position)
            y_position_max_e = max(y_position)
            y_position_min_e = min(y_position)
            position_num_e = len(x_position)
    
        x_position_max = max(bbox_x)
        x_position_min = min(bbox_x)
        y_position_max = max(bbox_y)
        y_position_min = min(bbox_y)        
               
        #在多边形的外围，需要过滤掉
        if ((y_position_max < y_position_min_e)or(y_position_min > y_position_max_e)
            or(x_position_max < x_position_min_e)or(x_position_min > x_position_max_e)):
            #print("box min max out of poly:", bbox_x, bbox_y)
            model_logger.info("box min max out of poly")
            return True
        
        #依次判断矩形框的每个点是否在多边形内
        Inpoly_rst=[0,0,0,0]
        for i in range(4):
            Inpoly_rst[i] = self.IsPtInPoly(bbox_x[i],bbox_y[i],x_position, y_position)
        
        '''CenterPt_rst = self.IsPtInPoly((bbox_x[0]+bbox_x[1])/2,(bbox_y[0]+bbox_y[3])/2)
        
        #四个顶点全部在多边形外，则在多边形外部，需要过滤掉
        if Inpoly_rst==[0,0,0,0]:
            if CenterPt_rst==0:
                print("box out of poly, center out of poly:", bbox_x, bbox_y)
                return True
            else:
                PolyArea = self.calcPolyArea(self.x_position,self.y_position)
                crossAreaRatios = PolyArea/((bbox_x[1]-bbox_x[0]) *(bbox_y[3] - bbox_y[0])+0.001)
                print("box out of poly, center in poly:", bbox_x, bbox_y)
                print("crossAreaRatios:", crossAreaRatios)
                if (crossAreaRatios < self.crossAreaRatios):
                    
                    return True
                else:
                    return False '''
                
        
        #四个顶点全部在多边形内，则在多边形内部，不需要过滤掉
        if Inpoly_rst==[1,1,1,1]:
            #print("all box in poly:", bbox_x, bbox_y)
            model_logger.info("all box in poly")
            return False 
        
        if (self.InPolyAreaRatios(bbox_x,bbox_y,Inpoly_rst,x_position, y_position) > self.crossAreaRatios):
            return False
        else:               
            return True         
    
    def IsPtInPoly(self,x,y,x_position, y_position):
        
        x_position_max_e = 0
        x_position_min_e = 0
        y_position_max_e = 0
        y_position_min_e = 0
        position_num_e = 0 
        
        if len(x_position) > 2 :
            x_position_max_e = max(x_position)
            x_position_min_e = min(x_position)
            y_position_max_e = max(y_position)
            y_position_min_e = min(y_position)
            position_num_e = len(x_position)
            
        if ((y < y_position_min_e)or(y > y_position_max_e)
            or(x < x_position_min_e)or(x > x_position_max_e)):
            return 0
        
        rst = 0        
        j = position_num_e -1
        for i in range(position_num_e):
            if ((y_position[i] > y) != (y_position[j] > y)) and (x < ((x_position[j]-x_position[i])*(y-y_position[i])/(y_position[j]-y_position[i]++0.00001) + x_position[i])):
                if rst == 0:
                    rst = 1
                else:
                    rst = 0
            j = i             
        return rst
     
    def calcPolyArea(self,x_pos,y_pos):
        PolyArea = 0
        
        if len(x_pos) < 3:
            return 0           
        
        j = len(x_pos) -1
        for i in range(len(x_pos)):            
            PolyArea = PolyArea +  (x_pos[i]*y_pos[j] - x_pos[j]*y_pos[i])
            j = i
            
        PolyArea = PolyArea/2.0 
        
        return abs(PolyArea)

     
    def InPolyAreaRatios(self,bbox_x, bbox_y, Inpoly_rst,x_position, y_position):       
        
        x_position_max_e = 0
        x_position_min_e = 0
        y_position_max_e = 0
        y_position_min_e = 0
        position_num_e = 0 
        
        if len(x_position) > 2 :
            x_position_max_e = max(x_position)
            x_position_min_e = min(x_position)
            y_position_max_e = max(y_position)
            y_position_min_e = min(y_position)
            position_num_e = len(x_position)
        
        #计算矩形和多边形的交点,该函数的矩形框必然是和多边形相交的
        #print("box out or cross Poly:", bbox_x,bbox_y)
        model_logger.info("box out or cross Poly: x:{}, y:{}".format(bbox_x,bbox_y))
        
        box_x =[bbox_x[0],bbox_x[1]]
        box_y =[bbox_y[0],bbox_y[3]]
        
        cross_info_lst=[]
        for x in box_x:        
            j = position_num_e -1        
            for i in range(position_num_e):
                if ((x_position[i] > x) != (x_position[j] > x)):                    
                    cross_y = (y_position[j]-y_position[i])*(x-x_position[i])/(x_position[j]-x_position[i]+0.00001)+ y_position[i]
                    if ((cross_y < box_y[1]) and (cross_y > box_y[0])):
                        cross_info = [x,int(cross_y)]
                        cross_info_lst.append(cross_info)
                j=i
        
        for y in box_y:        
            j = position_num_e -1        
            for i in range(position_num_e):
                if ((y_position[i] > y) != (y_position[j] > y)):
                    #print("y cross:",self.y_position[i],self.y_position[j],y)
                    cross_x = (x_position[j]-x_position[i])*(y-y_position[i])/(y_position[j]-y_position[i]+0.00001) + x_position[i]
                    #print(self.x_position[j],self.x_position[i],self.y_position[j],self.y_position[i])
                    #print("x cross:",cross_x,box_x[0], box_x[1])
                    if ((cross_x < box_x[1]) and (cross_x > box_x[0])):
                        cross_info = [int(cross_x), y]
                        cross_info_lst.append(cross_info)
                j=i
        
        #print("cross_info_lst",cross_info_lst)
        model_logger.info("cross_info_lst:{}".format(cross_info_lst))
        x_pos,y_pos = self.SortCrossPtAndBoxPt(cross_info_lst,bbox_x,bbox_y,Inpoly_rst,x_position, y_position) 
        #print("SortCrossPtAndBoxPt x_pos after:", x_pos)
        #print("SortCrossPtAndBoxPt y_pos after", y_pos)
        model_logger.info("SortCrossPtAndBoxPt x_pos after:{}".format(x_pos))
        model_logger.info("SortCrossPtAndBoxPt y_pos after:{}".format(y_pos))
        
        PolyArea = self.calcPolyArea(x_pos,y_pos)
        crossAreaRatios = PolyArea/((bbox_x[1]-bbox_x[0]) *(bbox_y[3] - bbox_y[0])+0.00001)
        #print("crossAreaRatios:", crossAreaRatios)
        model_logger.info("crossAreaRatios:{}".format(crossAreaRatios))        
        return crossAreaRatios        
        
        
    def SortCrossPtAndBoxPt(self, cross_info_lst,bbox_x,bbox_y,Inpoly_rst,x_position, y_position):
        x_pos=[]
        y_pos=[]
        
        #加入在多边形内的矩形框的顶点
        for i in range(4):
            if Inpoly_rst[i] == 1:
                x_pos.append(bbox_x[i])
                y_pos.append(bbox_y[i])        
        
        #加入矩形框和多边形的交点
        for cross_pt in cross_info_lst:                     
            x_pos.append(cross_pt[0])
            y_pos.append(cross_pt[1])             
        
        #加入在矩形框内的多边形的顶点
        for i in range(len(x_position)):            
            if (((x_position[i]< bbox_x[1]) and (x_position[i]> bbox_x[0])) 
                and((y_position[i]< bbox_y[3]) and (y_position[i]> bbox_y[0]))):                 
                x_pos.append(x_position[i])
                y_pos.append(y_position[i])
        
        #print("SortCrossPtAndBoxPt x_pos before:", x_pos)
        #print("SortCrossPtAndBoxPt y_pos before", y_pos)
        model_logger.info("SortCrossPtAndBoxPt x_pos before:{}".format(x_pos))
        model_logger.info("SortCrossPtAndBoxPt y_pos before:{}".format(y_pos))
        
        return self.ClockwiseSortPoints(x_pos,y_pos)
        
    def ClockwiseSortPoints(self, x_pos,y_pos):
        num = len(x_pos)
        #print("num:", num)
        if num < 3:
            return  x_pos,y_pos
        acc_x = 0
        acc_y = 0  
        for i in range(num):
            acc_x = acc_x + x_pos[i]
            acc_y = acc_y + y_pos[i]
        centerO = [acc_x/num, acc_y/num]
        
        
        for i in range(num):
            for j in range(num-i-1):
                if self.PointCmp([x_pos[j],y_pos[j]],[x_pos[j+1],y_pos[j+1]], centerO):                    
                    pt_tmp= [x_pos[j],y_pos[j]]
                    x_pos[j] = x_pos[j+1]
                    y_pos[j] = y_pos[j+1]
                    x_pos[j+1] = pt_tmp[0]
                    y_pos[j+1] = pt_tmp[1]
                    
            
        if self.PointCmp([x_pos[num-1],y_pos[num-1]],[x_pos[0],y_pos[0]], centerO):
            pt_tmp= [x_pos[0],y_pos[0]]
            x_pos[0] = x_pos[num-1]
            y_pos[0] = y_pos[num-1]
            x_pos[num-1] = pt_tmp[0]
            y_pos[num-1] = pt_tmp[1]
            
            for i in range(num):
                for j in range(num-i-1):
                    if self.PointCmp([x_pos[j],y_pos[j]],[x_pos[j+1],y_pos[j+1]], centerO):                    
                        pt_tmp= [x_pos[j],y_pos[j]]
                        x_pos[j] = x_pos[j+1]
                        y_pos[j] = y_pos[j+1]
                        x_pos[j+1] = pt_tmp[0]
                        y_pos[j+1] = pt_tmp[1]                              
                 
        return  x_pos,y_pos
    
    def PointCmp(self,pt1,pt2,centerO):
        #向量叉乘
        det = (pt1[0] - centerO[0])*(pt2[1] - centerO[1])- (pt1[1] - centerO[1])*(pt2[0] - centerO[0])
        
        if det < 0:
            return True
        if det > 0:
            return False
            
        #向量OA和向量OB共线，以距离判断大小
        d1 = pow((pt1[0] - centerO[0]),2)+pow((pt1[1] - centerO[1]),2)
        d2 = pow((pt2[0] - centerO[0]),2)+pow((pt2[1] - centerO[1]),2)
        
        return d1 > d2

    def drawInfo(self,bbox_x, bbox_y):
        cv2.rectangle(self.img,(bbox_x[0],bbox_y[0]),(bbox_x[1],bbox_y[3]),(255,0,255),2)
    
