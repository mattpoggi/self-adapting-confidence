import tensorflow as tf
import numpy as np


def conv2d(x, kernel_shape, strides=1, relu=True, padding='SAME'):
    W = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable("biases", kernel_shape[3], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
    return x

def encoding_unit(name, inputs, num_outputs):
    with tf.variable_scope('encoding' + str(name)):
        conv = tf.contrib.layers.conv2d(
                    inputs=inputs,
                    num_outputs=num_outputs,
                    kernel_size=3,
                    activation_fn=None
                )
        relu = tf.nn.relu(conv)
        pool = tf.contrib.layers.max_pool2d(relu, 2)

    forward = conv
    return pool, forward


def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def decoding_unit(number, inputs, num_outputs, forwards=None):
    with tf.variable_scope('decoding' + number):
        conv_transpose = tf.nn.relu(tf.contrib.layers.conv2d(upsample_nn(inputs, 2), num_outputs=num_outputs, kernel_size=3, activation_fn=None))

        if forwards is not None:
            if isinstance(forwards, (list, tuple)):
                for f in forwards:
                    conv_transpose = tf.concat([conv_transpose, f], axis=3)
            else:
                conv_transpose = tf.concat([conv_transpose, forwards], axis=3)
                
        conv = tf.contrib.layers.conv2d( 
                    inputs=conv_transpose,
                    num_outputs=num_outputs,
                    kernel_size=3,
                    activation_fn=None
                )

        relu = tf.nn.relu(conv)

    return relu
            

def pool_2d(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def pad(img, h, w):
    hpad = h-img.shape[1]
    wpad = w-img.shape[2]
    if hpad+wpad==0:
        return img, 0, 0
    else:
        return np.pad(img, ((0,0),(0,hpad),(0,wpad),(0,0)), 'constant'),hpad,wpad


def depad(img,hpad,wpad):
    return img[:,0:img.shape[1]-hpad,0:img.shape[2]-wpad,:]








def bilinear_sampler(imgs, coords):
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros_like(x_max)

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])

        return output


def build_coords(coords):
    height = coords.get_shape().as_list()[1]
    width = coords.get_shape().as_list()[2]
    pixel_coords = np.ones((1, height, width, 2))
    # build pixel coordinates and their disparity
    for i in range(0, height):
        for j in range(0, width):
            pixel_coords[0][i][j][0] = j
            pixel_coords[0][i][j][1] = i

    pixel_coords = tf.constant(pixel_coords, tf.float32)
    coords = tf.concat([coords, np.zeros((coords.get_shape().as_list()[0], height, width, 1))], axis=3)
    output = pixel_coords - coords

    return output


def generate_image_left( img, disp):
    coords = build_coords(disp)
    return bilinear_sampler(img, coords)


def generate_image_right( img, disp):
    coords = build_coords(-disp)
    return bilinear_sampler(img, coords)

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
    mu_y = tf.nn.avg_pool(y, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')

    sigma_x = tf.nn.avg_pool(x ** 2, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME') - mu_x ** 2
    sigma_y = tf.nn.avg_pool(y ** 2, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(x * y , [1, 3, 3, 1], [1, 1, 1, 1], 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def kitti_colormap(disparity, maxval=-1):
        """
        A utility function to reproduce KITTI fake colormap
        Arguments:
          - disparity: numpy float32 array of dimension HxW
          - maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)
        
        Returns a numpy uint8 array of shape HxWx3.
        """
        if maxval < 0:
                maxval = np.max(disparity)

        colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
        weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,5.747126436781609,8.771929824561404,5.405405405405405,8.771929824561404,0])
        cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])

        colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
        values = np.expand_dims(np.minimum(np.maximum(disparity/maxval, 0.), 1.), -1)
        bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum,axis=0),axis=0), disparity.shape[1], axis=1), disparity.shape[0], axis=0)
        diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
        index = np.argmax(diffs, axis=-1)-1

        w = 1-(values[:,:,0]-cumsum[index])*np.asarray(weights)[index]


        colored_disp[:,:,2] = (w*colormap[index][:,:,0] + (1.-w)*colormap[index+1][:,:,0])
        colored_disp[:,:,1] = (w*colormap[index][:,:,1] + (1.-w)*colormap[index+1][:,:,1])
        colored_disp[:,:,0] = (w*colormap[index][:,:,2] + (1.-w)*colormap[index+1][:,:,2])

        return (colored_disp*np.expand_dims((disparity>0),-1)*255).astype(np.uint8)


def uniqueness(disparity):
        disparity = (disparity[:,:,:,0]).astype(np.uint8)
        batch = disparity.shape[0]
        height = disparity.shape[1]
        width = disparity.shape[2]
        coords = np.stack([np.stack([ np.arange(b*width*height + y*width, b*width*height + y*width + width) for y in range(height)], 0) for b in range(batch)], 0) - disparity
        array = np.reshape(coords, batch*height*width)
        _, index, _, _ = np.unique(array, return_index=True,return_inverse=True,return_counts=True)
        array *= 0
        array[index] = 1
        return np.expand_dims(np.reshape(array, (batch, height, width)), -1).astype(np.float32)
        
def agreement(disparity, r, tau=1):
        disparity = (disparity[:,:,:,0]).astype(np.uint8)
        height = disparity.shape[1]
        width = disparity.shape[2]
        batch = disparity.shape[0]
        disparity = np.pad(disparity, ((0,0),(r,r),(r,r)), 'constant')
        wind = (r*2+1)
        neighbors = np.stack([disparity[:,k//wind:k//wind+height,k%wind:k%wind+width] for k in range(wind**2)], -1)
        neighbors = np.delete(neighbors, wind**2//2, axis=-1)
        template = np.stack([disparity[:,r:r+height,r:r+width]]*(wind**2), -1)
        template = np.delete(template, wind**2//2, axis=-1)
        agreement = (np.sum( np.abs(template - neighbors) < tau, axis=-1, keepdims=True ) ).astype(np.float32)

        return agreement
