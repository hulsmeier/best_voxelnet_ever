#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import glob
import argparse
import os
import time
import tensorflow as tf
import socket
import struct

from config import cfg
from model import RPN3D

from utils import *
from utils.kitti_loader import iterate_data, sample_test_data



parser = argparse.ArgumentParser(description='testing')
parser.add_argument('-d', '--decrease', type=bool, nargs='?', default=False,
                    help='set the flag to True if decrease model')
parser.add_argument('-m', '--minimize', type=bool, nargs='?', default=False,
                    help='set the flag to True if minimize model')
args = parser.parse_args()


res_dir = os.path.join('.', './predictions')
save_model_dir = os.path.join('.', 'save_model', 'default')
    
os.makedirs(res_dir, exist_ok=True)
os.makedirs(os.path.join(res_dir, 'data'), exist_ok=True)

serverIPAddress = '127.0.0.1'
serverPortNumber = 44444

def main(_):
    
    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
            visible_device_list=cfg.GPU_AVAILABLE,
            allow_growth=True
        )

        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={"GPU": cfg.GPU_USE_COUNT,},
            allow_soft_placement=True,
        )

        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                decrease=args.decrease,
                minimize=args.minimize,
                single_batch_size=1,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            
            # param init/restore
            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))


            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientSocket:

                connectedToServer = False

                while connectedToServer == False:
                    try:
                        clientSocket.connect((serverIPAddress, serverPortNumber))
                        connectedToServer = True
                    except Exception as e:
                        print("Waiting for server...")
                
                while True:

                    print("Waiting for data...")
                    numberOfBytesReceived = 0
                    messageSizeReceived = False
                    currentMessageSize = 0

                    bytesFromClient = b''

                    newBytesFromClient = clientSocket.recv(1000024)

                    while len(newBytesFromClient) > 0:

                        numberOfBytesReceived += len(newBytesFromClient)
                        bytesFromClient += newBytesFromClient

                        if(bytesFromClient == b''):
                            break


                        if messageSizeReceived == False:
                            if numberOfBytesReceived >= 4:
                                currentMessageSize = struct.unpack('i', bytesFromClient[:4])[0]
                                messageSizeReceived = True
                            else:
                                continue


                        if numberOfBytesReceived >= currentMessageSize + 4:
                            
                            f_lidar = bytesFromClient[4:]

                            batch = sample_data_live(f_lidar)

                            results = model.predict_step_live(sess, batch)
            
                            #for result in zip(results):
                            #    labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                            #    print('write out {} objects'.format(len(labels)))

                            print('write out {} objects'.format(len(results)))

                            break

                        newBytesFromClient = clientSocket.recv(1000024)

if __name__ == '__main__':
    tf.app.run(main)
