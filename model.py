from dataloader import Dataloader
import tensorflow as tf
import numpy as np
import ops
import time
import os
import cv2
import progressbar
import sys

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


class OTB(object):

    def __init__(self, sess, initial_learning_rate, image_height, image_width, mode, dataset, left_dir, right_dir, disp_dir, p, q, colors):
                                  
        self.sess = sess
        self.mode = mode
        self.colors = colors

        self.image_height = image_height        
        self.image_width = image_width

        self.initial_learning_rate = initial_learning_rate
        self.p = p
        self.q = q

        # build data pipeline
        self.dataset = dataset
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir
        self.dataloader = Dataloader(dataset=self.dataset, left_dir=self.left_dir, right_dir=self.right_dir, disp_dir=self.disp_dir)
        self.placeholders = {'left':tf.placeholder(tf.float32, [1, self.image_height, self.image_width, 3], name='left'),
                             'right':tf.placeholder(tf.float32, [1, self.image_height, self.image_width, 3], name='right'),
                             'disp':tf.placeholder(tf.float32, [1, self.image_height, self.image_width, 1], name='disparity')}
        self.learning_rate = tf.placeholder(tf.float32, shape=[])        
        self.ConfNet_v2() 
        if self.mode == 'otb-online': 
            self.build_losses()

    def ConfNet_v2(self):
        print(" [*] Building ConfNet model...")

        kernel_size = 3
        filters = 32
        disp = self.placeholders['disp']

        with tf.variable_scope('ConfNet'):
                                    
            with tf.variable_scope('disparity'):  
                with tf.variable_scope("conv1"):
                    self.conv1_disparity = ops.conv2d(disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='SAME')
            
            model_input = self.conv1_disparity
            self.net1, self.scale1 = ops.encoding_unit('1', model_input, filters)

            self.net2, self.scale2 = ops.encoding_unit('2', self.net1,   filters * 4)
            self.net3, self.scale3 = ops.encoding_unit('3', self.net2,   filters * 8)
            self.net4, self.scale4 = ops.encoding_unit('4', self.net3,   filters * 16)
            self.net5 = ops.decoding_unit('4', self.net4, num_outputs=filters * 8, forwards=self.scale4)
            self.net6 = ops.decoding_unit('3', self.net5, num_outputs=filters * 4, forwards=self.scale3)
            self.net7 = ops.decoding_unit('2', self.net6, num_outputs=filters * 2,  forwards=self.scale2)
            self.net8 = ops.decoding_unit('1', self.net7, num_outputs=filters, forwards=self.conv1_disparity)
                        
            self.prediction = ops.conv2d(self.net8, [kernel_size, kernel_size, filters, 1], 1, False, padding='SAME')


    def build_losses(self):
        with tf.variable_scope('loss'):

            # prepare validity mask
            self.valid = tf.ones_like(self.placeholders['disp'])

            # texture mask
            self.warped = ops.generate_image_left(self.placeholders['right'], self.placeholders['disp'])
            self.reprojection = tf.reduce_sum(0.85*ops.SSIM(self.warped, self.placeholders['left']) + 0.15*tf.abs(self.warped - self.placeholders['left']), -1, keepdims=True)
            self.identity = tf.reduce_sum(0.85*ops.SSIM(self.placeholders['left'], self.placeholders['right']) + 0.15*tf.abs(self.placeholders['left'] - self.placeholders['right']), -1, keepdims=True)                        
            self.t = tf.cast(self.identity > self.reprojection, tf.float32)
            
            # agreement mask
            self.a = tf.cast(tf.py_func(ops.agreement, [self.placeholders['disp'], 2], tf.float32) > (5**2-1)*0.5, tf.float32)

            # uniqueness mask
            self.u = tf.py_func(ops.uniqueness, [self.placeholders['disp']], tf.float32)

            # initializing inliers and outliers masks
            self.P = tf.ones_like(self.placeholders['disp'])
            self.Q = tf.ones_like(self.placeholders['disp'])

            if 't' in self.p:
                self.P *= self.t
            if 'a' in self.p:
                self.P *= self.a
            if 'u' in self.p:
                self.P *= self.u

            if 't' in self.q:
                self.Q *= (1-self.t)
            if 'a' in self.q:
                self.Q *= (1-self.a)
            if 'u' in self.q:
                self.Q *= (1-self.u)

            # quick implementation for MBCE
            self.proxysignal = self.P * (1 - self.Q)
            self.valid = self.valid * (self.P + self.Q) 
            self.loss = tf.losses.sigmoid_cross_entropy(self.proxysignal, self.prediction, self.valid)

    def run(self, args):
        print(" [*] Running....")

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        disp_batch = self.dataloader.disp
        left_batch = self.dataloader.left
        right_batch = self.dataloader.right
        image_name = self.dataloader.filename
        num_samples = count_text_lines(self.dataset)

        if self.mode == 'reprojection':
            print(' [*] Generating Reprojection error')
            warped = ops.generate_image_left(self.placeholders['right'], self.placeholders['disp'])
            net_output = -(tf.reduce_sum(0.85*ops.SSIM(warped, self.placeholders['left']) + 0.15*tf.abs(warped - self.placeholders['left']), -1, keepdims=True))

        elif self.mode == 'agreement':
            print(' [*] Generating Disparity Agreement')
            net_output = tf.py_func(ops.agreement, [self.placeholders['disp'], 2], tf.float32)

        elif self.mode == 'uniqueness':
            print(' [*] Generating Uniqueness Constraint')
            net_output = tf.py_func(ops.uniqueness, [self.placeholders['disp']], tf.float32)

        elif 'otb' in self.mode:

            print(' [*] OTB inference')
            self.saver = tf.train.Saver()
            if args.checkpoint_path:
                self.saver.restore(self.sess, args.checkpoint_path)
                print(" [*] Load model: SUCCESS")
            else:
                print(" [*] Load failed...neglected")
                print(" [*] End Testing...")
                raise ValueError('args.checkpoint_path is None')

            net_output = tf.nn.sigmoid(self.prediction)

            if self.mode == 'otb-online':
                print(' [*] Online Adaptation')
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=tf.global_variables())
        else:
            print(" [*] Unsupported testing mode!")
            raise ValueError('args.mode is not supported')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(" [*] Start Testing...")
        bar = progressbar.ProgressBar(max_value=num_samples)
        for step in range(num_samples):
            batch_left, batch_right, batch_disp, filename = self.sess.run([left_batch, right_batch, disp_batch, image_name])

            val_disp, hpad, wpad = ops.pad(batch_disp, self.image_height, self.image_width)
            val_left, _, _ = ops.pad(batch_left, self.image_height, self.image_width)
            val_right, _, _ = ops.pad(batch_right, self.image_height, self.image_width)

            if self.mode == 'otb-online':
                _, confidence = self.sess.run([self.optimizer, net_output], feed_dict={self.placeholders['disp']: val_disp, self.placeholders['left']: val_left, self.placeholders['right']: val_right, self.learning_rate: self.initial_learning_rate})
            else:
                confidence = self.sess.run(net_output, feed_dict={self.placeholders['disp']: val_disp, self.placeholders['left']: val_left, self.placeholders['right']: val_right})

            confidence = ops.depad(confidence, hpad, wpad)

            outdir = args.output_path + '/' + filename
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            confidence_file = outdir + '/'+self.mode+'.png'
            c = confidence[0]
            c = (c - np.min(c)) / (np.max(c) - np.min(c))
            cv2.imwrite(confidence_file, (c * (2**16-1)).astype('uint16'))
            if self.colors:
                color_file = outdir + '/'+self.mode+'-color.png'
                cv2.imwrite(color_file, cv2.applyColorMap(((1-c)*(2**8-1)).astype('uint8'),cv2.COLORMAP_WINTER))

            bar.update(step+1)

        coord.request_stop()
        coord.join(threads)
