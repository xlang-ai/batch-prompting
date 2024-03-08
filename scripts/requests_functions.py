import boto3
import threading
import time
from time import sleep
import numpy as np
from datetime import datetime
import sys
import base64
import json
import random
import math

max_req = {}
my_batches = {}

#mylambda = boto3.client('lambda')
allocated_memory = 3008
my_service_time = {}


class MyRequest():

    def __init__(self, arrival_time=0, end_time=0, latency=0, request_size=0, global_time=0.0):
        self.arrival_time = arrival_time
        self.end_time = end_time
        self.latency = end_time
        self.wait_time = 0
        self.request_size = request_size
        self.global_time = global_time

    def my_print(self):
        print("start = {} end = {} latency =- {}".format(self.arrival_time, self.end_time, self.latency))


class MyServerless(threading.Thread):

    def __init__(self, batch_size, queue_time, request_list, actual_batch_size, inter_arrival, time_out, request_class,
                 total_classes, max_batch_req, interval, batch_content):
        threading.Thread.__init__(self)
        self.batch_size = batch_size
        self.queue_time = queue_time
        self.request_list = request_list
        self.actual_batch_size = actual_batch_size
        self.time_out = time_out
        self.inter_arrival = inter_arrival
        self.request_class = request_class
        self.max_batch_req = max_batch_req
        self.total_classes = total_classes
        self.interval = interval
        self.batch_content = batch_content
        self.file_size = ["12", "24", "28", "32", "36", "42", "44", "48", "52", "56", "60", "64", \
                          "68", "72", "76", "80", "84", "88", "92", "96", "100", "104", "108", "112", "116", "120",
                          "124", \
                          "128", "132", "137", "140", "144"]
        self.memroy = {1024: 0, 1152: 1, 1920: 7, 2688: 13, 2048: 8, 2304: 10, 2176: 9, 2944: 15, 1280: 2, \
                       2560: 12, 1408: 3, 1536: 4, 2816: 14, 1664: 5, 2432: 11, 1792: 6}
        self.class_service_time = my_service_time  # self.read_service_time()

    def read_service_time(self):
        service_time = {}
        with open("service_time.json", "r") as fp:
            service_time_json = json.load(fp)
            service_time = service_time_json["service_time"]
        return service_time[allocated_memory]
        '''outer = []
        for i in range(len(self.class_service_time)):
            inner = []
            for j in range(len(self.class_service_time[0])):
                y = (float('{:.4f}'.format(self.class_service_time[i][j])))
                #print(inner)
                inner.append(y)
            print(inner)
            outer.append(inner)
        print(outer)    	'''

    def send_request(self):
        file_size = [12, 24, 28, 32, 36, 42, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                     116, 120, 124, 128, 132, 137, 140, 144]
        file_lambda_logs = "Lambda_logs_batch_{}_interval_{}_{}_time_out_{}_class_{}.log".format(self.actual_batch_size,
                                                                                                 self.inteval - 1,
                                                                                                 self.interval,
                                                                                                 self.time_out,
                                                                                                 self.request_class)

        mutex_lock2 = threading.Lock()

        data = {"batch_size": self.batch_size, "file_name": file_size[self.max_batch_req]}
        batch_service_time = mylambda.invoke(FunctionName='deep_speech', InvocationType='RequestResponse',
                                             LogType='Tail', Payload=json.dumps(data))

        mutex_lock2.acquire()

        with open(file_lambda_logs, "a+") as fl:
            fl.write("{}\n".format(base64.b64decode(batch_service_time['LogResult'])))
        fl.close()

        mutex_lock2.release()

        return batch_service_time['Payload'].read()

    def run(self):
        file_batch_latency = "Latency_perbatch_batch_{}_interval_{}_{}_time_out{}_memory_{}_class_{}.log".format(
            self.actual_batch_size, \
            self.interval - 1, self.interval, self.time_out, allocated_memory, self.request_class)
        file_request_latency = "Latency_per_request_batch_{}_interval_{}_{}_time_out{}_memory_{}_class_{}.log".format(
            self.actual_batch_size, \
            self.interval - 1, self.interval, self.time_out, allocated_memory, self.request_class)

        mutex_1 = threading.Lock()
        mutex_2 = threading.Lock()
        indexes = math.ceil((len(self.class_service_time) / float(self.total_classes)))
        class_indx = int(min((self.request_class + 1) * indexes - 1, len(self.class_service_time) - 1))
        # batch_latency = float(self.send_request())
        batch_latency = self.class_service_time[self.max_batch_req][self.batch_size - 1]
        # print("index {}, latency {}".format(class_indx,self.class_service_time[class_indx][self.batch_size-1]))
        mutex_1.acquire()
        with open(file_batch_latency, "a+") as fl:
            fl.write(
                "batch\t{}\tlatency\t{}\tmax_req\t{}\tcontent\t{}\ttime\t{}\n".format(self.batch_size, batch_latency,
                                                                                      self.max_batch_req, \
                                                                                      self.batch_content,
                                                                                      self.request_list[
                                                                                          -1].global_time))
        fl.close()
        mutex_1.release()

        with open(file_request_latency, "a+") as ft:
            for request in self.request_list:
                request.latency = request.wait_time + float(batch_latency)
                mutex_2.acquire()
                ft.write("Request_latency\t{}\tbatch_size\t{}\trequest_Size\t{}\ttime\t{}\n".format(request.latency,
                                                                                                    self.batch_size,
                                                                                                    request.request_size,
                                                                                                    request.global_time))
                # print("Request_latency\t{}\tbatch_size\t{}\trequest_Size\t{}\twait\t{}\n".format(request.latency,self.batch_size,request.request_size, request.wait_time))
                mutex_2.release()
        ft.close()


def read_twitter_arrival(stream_id):
    my_arrival = []
    count = 0  # workload
    trace_filepath = './buffer{}/interArrPerClass_1_buff/twitter_buff_{}_7_8.log'.format(stream_id, stream_id)
    print(trace_filepath)
    # with open('./workload/Twitter_{}-{}_32_bimodal'.format(stream_id-1, stream_id), 'r') as fp:
    # with open('./workload/Twitter_0-1_32_bimodal'.format(stream_id-1, stream_id), 'r') as fp:
    # with open('../custom_traces/len_32_max_8_bimodal/Twitter_{}-{}_32_len_32_max_8_bimodal'.format(stream_id-1, stream_id), 'r') as fp:
    # with open('../Twitter_arrival_with_reqid/Twitter_{}-{}_32'.format(stream_id-1, stream_id), 'r') as fp:
    with open(trace_filepath, 'r') as fp:
        for line in fp:
            val = line.strip('\n')
            val = val.split()
            my_arrival.append([float(val[0]), int(val[1])])
            count += 1
        # if count >=250000:
        #	break
    return my_arrival


def generate_classes(number_of_classes, time_outs, batch_sizes, start, my_end):
    myclass = dict()
    for i in range(start, my_end):
        myclass[i] = {}
        myclass[i]["request_class"] = i
        myclass[i]["queue"] = []
        myclass[i]["request_count"] = 0
        myclass[i]["total_delay"] = 0.0
        myclass[i]["time_out"] = time_outs[i]
        myclass[i]["batch_size"] = batch_sizes[i]
        myclass[i]['bs'] = []
        myclass[i]['max'] = 0
        myclass[i]['buffer_ticks'] = float(0.0)
    return myclass


def preparing_requests(request_class, time_out, batch_size, req_inter_arrival, num_classes, interval):
    batch_com = ''
    if request_class["request_count"] > 0 and request_class["request_count"] < batch_size and \
            request_class["total_delay"] + req_inter_arrival <= float(time_out):
        # sleep(req_inter_arrival)
        # print("batch\t{}delay\t{}".format(request_class["request_count"], request_class["total_delay"]))
        request_class["total_delay"] += req_inter_arrival
    elif request_class["request_count"] == batch_size:
        # print(request_class["request_count"])
        # print("batch_formed \tbatch\t{}\tdelay\t{}".format(request_class["request_count"],request_class["total_delay"]))
        request_class["bs"].append(request_class["request_count"])
        request_list = []
        batch_content = []
        while len(request_class["queue"]) > 0:
            request = request_class["queue"].pop(0)
            batch_com = batch_com + str(request.request_size)
            request.wait_time = (request_class["total_delay"]) - request.arrival_time
            request_list.append(request)  # *1000
            batch_content.append(request.request_size)
        if len(request_list) > 0:
            b_content = ','.join([str(v) for v in batch_content])
            MyServerless(request_class["request_count"], float(request_class["total_delay"]),
                         request_list, batch_size, inter_arrival, time_out, request_class["request_class"],
                         num_classes, request_class["max"], interval, b_content).start()  # *1000
            max_req[request_class["max"]] = max_req.get(request_class["max"], 0) + 1
            my_key = ''.join(sorted(batch_com))
            # my_batches[batch_com] = my_batches.get(batch_com, 0)+1
            my_batches[my_key] = my_batches.get(my_key, 0) + 1
            request_class["request_count"] = 0
            request_class["total_delay"] = 0.0
            request_class["max"] = 0
        # sleep(req_inter_arrival)
    elif (request_class["total_delay"] + req_inter_arrival >= time_out and request_class["request_count"] > 0):
        # print(request_class["request_count"])
        # sleep(time_out - request_class["total_delay"])
        # print("timeout_happend\tbatch\t{}delay\t{}".format(request_class["request_count"], request_class["total_delay"]))
        request_class["total_delay"] = float(time_out)
        delayRemainder = req_inter_arrival - (time_out - request_class["total_delay"])
        request_class["bs"].append(request_class["request_count"])
        request_list = []
        batch_content = []
        while len(request_class["queue"]) > 0:
            request = request_class["queue"].pop(0)
            batch_com = batch_com + str(request.request_size)
            request.wait_time = (request_class["total_delay"]) - request.arrival_time
            request_list.append(request)  # *1000
            batch_content.append(request.request_size)
        if len(request_list) > 0:
            b_content = ','.join([str(v) for v in batch_content])
            MyServerless(request_class["request_count"], float(request_class["total_delay"]),
                         request_list, batch_size, inter_arrival, time_out, request_class["request_class"],
                         num_classes, request_class["max"], interval, b_content).start()  # *1000
            max_req[request_class["max"]] = max_req.get(request_class["max"], 0) + 1
            my_key = ''.join(sorted(batch_com))
            # my_batches[batch_com] = my_batches.get(batch_com, 0)+1
            my_batches[my_key] = my_batches.get(my_key, 0) + 1
            request_class["request_count"] = 0
            request_class["total_delay"] = 0.0
            request_class["max"] = 0
        # sleep(delayRemainder)


def write_arrival_per_buffer(data, class_id, timeout, batch, interval):
    filename = "global_arrival_buffer_{}_batch_{}_timeout_{}".format(class_id, batch, timeout)
    with open(filename, 'w') as fp:
        fp.write("Inter_Arrival\trequest_id\tbuffer_id\n")
        for val in data:
            fp.write("{}\t{}\t{}\n".format(val[0], val[1], val[2]))


def generate_request(inter_arrival, num_classes, num_requests, class_id, t_out, interval, b_size):
    num_buffers = int(sys.argv[1])
    with open('requests_and_loads_random', 'r') as fp:
        chuncks_probs = json.load(fp)
    num_classes = num_classes
    route_config = chuncks_probs[str(num_classes)]['requests']
    # route_config = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 8], [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 16], [16, 17, 18, 19], [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
    start_req_id = route_config[class_id][0]  # (num_requests//num_classes)*class_id
    end_req_id = route_config[class_id][-1] + 1  # min(num_requests, start_req_id+ (num_requests//num_classes))
    number_request_types = num_requests  # number of request types 12KB to 120KB
    print(start_req_id, '\t', end_req_id)
    time_outs = [t_out * 1000] * 32  # ,0.5,0.5,0.5,0.5,0.5,0.5,0.5]#, 0.5, 1.0, 0.3]
    batch_sizes = [b_size] * 32  # ,10,10,10,10,10,10,10]
    request_classes = generate_classes(num_classes, time_outs, batch_sizes, class_id, class_id + 1)

    global_arrival_process = read_twitter_arrival(class_id)
    total_requests = len(global_arrival_process)
    test = {}
    test1 = {}
    global_timer = 0.0
    print("Done reading arrival process total requests {}".format(total_requests))
    print(request_classes)
    for i in range(0, total_requests):  # total_requests
        arrival_request = global_arrival_process[i][1]
        arrival_class = class_id  # global_arrival_process[i][2]
        test1[arrival_request] = test1.get(arrival_request, 0) + 1

        # if arrival_request in request_classes:
        if start_req_id <= arrival_request and arrival_request < end_req_id:
            # print("{}\t{}".format(request_classes[arrival_class]["total_delay"],request_classes[arrival_class]["request_count"]))
            total_delay = request_classes[arrival_class]["total_delay"]
            test[arrival_request] = test.get(arrival_request, 0) + 1
            request_classes[arrival_class]["queue"].append(
                MyRequest((total_delay), 0, 0, arrival_request, global_timer))
            request_classes[arrival_class]["max"] = max(arrival_request, request_classes[arrival_class]["max"])
            request_classes[arrival_class]["request_count"] += 1
        buffer_inter_arrival = global_arrival_process[i][0] * 1000
        global_timer += buffer_inter_arrival
        # print("inter_arrival\t{}".format(buffer_inter_arrival))
        preparing_requests(request_classes[arrival_class], request_classes[arrival_class]["time_out"],
                           request_classes[arrival_class]["batch_size"], buffer_inter_arrival, num_classes, interval)
    print(test)
    msum = 0.0
    for k in test:
        msum += test[k]
    print(test1)
    print(msum)  # /float(total_requests))

    msum = 0.0
    for k in test1:
        msum += test1[k]
    print(msum)
    print("Class {} mean batch size {}".format(class_id, np.mean(request_classes[class_id]["bs"])))


def read_service_time():
    service_time = {}
    with open("service_time_256.json", "r") as fp:
        service_time_json = json.load(fp)
        service_time = service_time_json["service_time"]
    if allocated_memory not in service_time:
        print("Memory value not in the file")
        if int(allocated_memory) < 1024:
            return service_time[str(1024)]
        val = int(allocated_memory) + 64
        return service_time[str(val)]
    return service_time[allocated_memory]


if __name__ == '__main__':

    print("This is my start time")
    inter_arrivals = [1.0]  # , 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]
    num_classes = int(sys.argv[1])  # number of buffers
    num_requests = 32  # int(sys.argv[2])
    class_id = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    t_out = float(sys.argv[4])
    interval = int(sys.argv[5])
    allocated_memory = (sys.argv[6])
    my_service_time = read_service_time()
    print(t_out)
    # generate_request(inter_arrivals[0],  num_classes, num_requests, class_id, t_out)
    for inter_arrival in inter_arrivals:
        # for batch in batch_size:
        #   for tout in time_out:
        # response = mylambda.update_function_configuration(FunctionName = 'DeepLearning_Lambda_new', MemorySize=3008, Environment = {'Variables' : {"model_bucket_name" : "deeplearning", "region" : "us-east-1", "BS" : "{}".format(batch), "delay" :"{}".format(0)}})
        generate_request(inter_arrival, num_classes, num_requests, class_id, t_out, interval, batch_size)
        # sleep(10)
