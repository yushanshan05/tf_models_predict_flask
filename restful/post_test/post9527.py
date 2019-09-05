#coding=utf-8
import requests
import time
import json
#import base64
#import cv2
import os
from PIL import Image,ImageDraw,ImageFont
#from scipy import misc

num = 10000000000000000
mtcnn_elapsed = 0
facenet_elapsed = 0
emotion_elapsed = 0
eye_elapsed = 0
angle_elapsed = 0
alltime = 0

i = 0
sum_time = 0
start = time.time()
for i in xrange(num):
    #start = time.clock()
    s = requests
    
    imagepath = '/opt/models/object_detection/test_images'
    imagepath_post = '/opt/models/object_detection/test_images'
    imagepath_out = '/opt/models/object_detection/test_images/out'
    files= os.listdir(imagepath)
    for file in files:
        if file.endswith('.jpg'):
            print(file)
            time.sleep(1)
            image = os.path.join(imagepath_post,file)
            data={"data":image}
            my_json_data = json.dumps(data)
            headers = {'Content-Type': 'application/json'}
            single_start = time.time()
            r = s.post('http://localhost:8080/user', headers=headers,data = my_json_data,)
            #r = s.post('http://localhost:8081/user', headers=headers,data = my_json_data,)
            single_end = time.time() - single_start
            
            print (i)
            print ('single_time:{}'.format(single_end))
            i = i+1
            sum_time = sum_time + single_end
            print ('average_time:{}'.format(sum_time/i))
            print('port:8080')
            continue
            #add plot
            '''
            #cv2
            img = cv2.imread(os.path.join(imagepath,file))
            data= {}
            print type(r.json())
            data = r.json()
            datalist = []
            datalist = data['data']
            print(len(datalist))
            print(datalist)
            continue
            
            for j in xrange(len(datalist)):
                singledata = {}
                boxdict = {}
                singledata = datalist[j]
                boxdict = singledata['bbox']
                xmin = boxdict['xmin']
                ymin = boxdict['ymin']
                xmax = boxdict['xmax']
                ymax = boxdict['ymax']
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,255,0))
                
                font= cv2.FONT_HERSHEY_SIMPLEX
                strname = singledata['cls']
                strscore = singledata['score']
                #print (type(strscore))
                print (strscore)
                cv2.putText(img, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin-10), font, 1,(0,0,255),2)
            print(os.path.join(imagepath_out,file))
            cv2.imwrite(os.path.join(imagepath_out,file), img)
            '''
            
            #pil
            img = Image.open(os.path.join(imagepath,file))
            data= {}
            #print type(r.json())
            data = r.json()
            datalist = []
            datalist = data['data']
            #print(len(datalist))
            print(datalist)
            #continue
            draw = ImageDraw.Draw(img)
            for j in xrange(len(datalist)):
                singledata = {}
                boxdict = {}
                singledata = datalist[j]
                boxdict = singledata['bbox']
                xmin = boxdict['xmin']
                ymin = boxdict['ymin']
                xmax = boxdict['xmax']
                ymax = boxdict['ymax']
                #cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,255,0))
                draw.line((xmin, ymin,xmax, ymin),fill=128)
                draw.line((xmax, ymin,xmax, ymax),fill=128)
                draw.line((xmin, ymax,xmax, ymax),fill=128)
                draw.line((xmin, ymin,xmin, ymax),fill=128)
                #font= cv2.FONT_HERSHEY_SIMPLEX
                strname = singledata['cls']
                strscore = singledata['score']
                #print (type(strscore))
                #print (strscore)
                draw.text((xmin, ymin-20), strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', font=ImageFont.truetype('./output/1.ttf', 20), fill="#ff0000")
                #cv2.putText(img, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin-10), font, 1,(0,0,255),2)
            print(os.path.join(imagepath_out,file))

            img.save(os.path.join(imagepath_out,file))
end = time.time() - start
print (end)

#plot
#imagepath = '/data/ligang/detectron/Detectron-master/restful/vis/806_180507070134.jpg'
#img = cv2.imread(imagepath)
#cv2.rectangle(img, (136,63), (765,474),3)
#cv2.rectangle(img, (130,50), (537,239),3)
#cv2.imwrite('./001_new.jpg', img)
'''
################################################################
############################# curl #############################
curl -X POST 'http://192.168.151.32:9527/user' -d '{"data":"/opt/ligang/Detectron/restful/images/20180828_1C1B0D24D586_00185.jpg"}' -H 'Content-Type: application/json'


curl -X POST 'http://192.168.200.213:9527/user' -d '{"data":"https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1526895699811&di=5ce6acbcfe8f1d93fe65d3ae8eb3287d&imgtype=0&src=http%3A%2F%2Fimg1.fblife.com%2Fattachments1%2Fday_130616%2F20130616_e4c0b7ad123ca263d1fcCnkYLFk97ynn.jpg.thumb.jpg"}' -H 'Content-Type: application/json'
'''
