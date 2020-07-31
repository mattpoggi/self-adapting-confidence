import tensorflow as tf

class Dataloader(object):

    def __init__(self, dataset, left_dir, right_dir, disp_dir):

        self.dataset = dataset
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir

        self.left = None
        self.right = None
        self.disp  = None

        input_queue = tf.train.string_input_producer([self.dataset], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line], '.').values

        self.left = tf.stack([tf.cast(self.read_image(tf.string_join([self.left_dir, line]), [None, None, 3]), tf.float32)], 0)
        self.right = tf.stack([tf.cast(self.read_image(tf.string_join([self.right_dir, line]), [None, None, 3]), tf.float32)], 0)
        self.disp = tf.stack([tf.cast(self.read_image(tf.string_join([self.disp_dir, split_line[0], '.png']), [None, None, 1], dtype=tf.uint16), tf.float32)], 0) / 256.
        self.filename = split_line[0]

    def read_image(self, image_path, shape=None, dtype=tf.uint8, norm=False):
        image_raw = tf.read_file(image_path)
        if dtype == tf.uint8:
            image = tf.image.decode_image(image_raw)
        else:
            image = tf.image.decode_png(image_raw, dtype=dtype)
        if shape is None:
            image.set_shape([None, None, 3])
        else:
            image.set_shape(shape)

        return image
