#coding=utf8
'''
random.randint(a, b):用于生成一个指定范围内的整数。
其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b

random.choice(sequence)：从序列中获取一个随机元素
参数sequence表示一个有序类型（列表，元组，字符串）

'''
import httplib,json
import time
import threading
from random import randint,choice 
import os
import requests
#创建请求函数
def postRequest(threadNum):
    '''
    postJson={                     
                }
    #定义需要进行发送的数据
    postData=json.dumps(postJson)

    #定义一些文件头
    headerdata = {
      
        "content-type":"application/json",
         }
    
    #接口
    requrl ="/v1/query"
    
    #请求服务,例如：www.baidu.com
    hostServer=""
    #连接服务器       
    conn = httplib.HTTPConnection(hostServer)
    #发送请求       
    conn.request(method="POST",url=requrl,body=postData,headers=headerdata)
    
    #获取请求响应       
    response=conn.getresponse()
    #打印请求状态
    if response.status in range(200,300):
        print u"线程"+str(threadNum)+u"状态码："+str(response.status)       
    conn.close()   
    '''
    file = '20180928_1C1B0D228AF1_00007.jpg'
    imagepath_post = '/opt/models/object_detection/test_images'
    image = os.path.join(imagepath_post,file)
    data={"data":image}
    my_json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    single_start = time.time()
    stime = time.localtime(single_start)
    poststartime=str(time.strftime("%Y%m%d%H%M%S",stime))
    data_secs = (single_start - int(single_start)) * 1000
    poststartime_sec = poststartime + str("%03d" % data_secs)
    print('poststart-time:{}'.format(poststartime_sec))
    s = requests
    r = s.post('http://localhost:8080/user', headers=headers,data = my_json_data,)
    single_end = time.time() - single_start
    print ('cam:{}-threas:{}-time:{}'.format(6,threadNum,single_end))
    
    
    

def run(threadNum,internTime,duration):   
    #创建数组存放线程    
    threads=[] 
    try:
        #创建线程
        for i in range(1,threadNum):
            #针对函数创建线程  
            t=threading.Thread(target=postRequest,args=(i,))
            #把创建的线程加入线程组     
            threads.append(t)  
    except Exception,e:
        print e
        
    try: 
        #启动线程  
        for thread in threads: 
                thread.setDaemon(True)
                thread.start()
                time.sleep(internTime) 
                        
        #等待所有线程结束
        for thread in  threads: 
                thread.join(duration)                                          
    except Exception,e:
            print e

if __name__ == '__main__':
    startime=time.strftime("%Y%m%d%H%M%S")

    now=time.strftime("%Y%m%d%H%M%S")
    #duration=raw_input(u"输入持续运行时间:")
    duration = 3000000  #00(hour)00(min)00(sec)  eg. 000010 - duration 10 sec
    #print(str(int(startime)+duration))
    #print('now:{}'.format(now))
    #while (startime+str(duration))!=now: 
    while (int(startime)+duration)>int(now):
        run(10,1,int(duration))
        now=time.strftime("%Y%m%d%H%M%S") 
        print('start:{}'.format(startime))
        print('now:{}'.format(now))
