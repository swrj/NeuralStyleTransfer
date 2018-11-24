import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfeager
from tensorflow.contrib.learn.python.learn import trainable
import tensorflow.contrib.eager as tensor_eager
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


"""
 we use eager execution because we don't want to deal tf variable and etc.
 because without eager execution tree based computations are disabled
 and it kind of gives normal results as though compiled using a normal
 python interpreter 
 
"""
def load_img_preprocess(image_path):
    img_str = tf.read_file(image_path)

    img_decode = tf.image.decode_jpeg(img_str, 3)

    img = tf.cast(img_decode, tf.float32)

    dim =512.0

    height = tf.to_float(tf.shape(img)[1])

    width = tf.to_float(tf.shape(img)[0])

    scale = tf.cond(tf.greater(height, width), lambda: dim/width, lambda: dim/height)

    newHeight = tf.to_int32(height * scale)
    newWidth = tf.to_int32(width * scale)

    img = tf.image.resize_images(img, [newHeight, newWidth])

    """VGG_MEAN = [123.68, 116.78, 103.94]  # This is R-G-B for Imagenet

    img = tf.random_crop(img, [224, 224, 3])
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    img = img - means
    """
    img = np.expand_dims(img, axis=0)

    VGG_MEAN = [123.68, 116.78, 103.94]

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    img = img - means

    return img


def restore_image(processed_image):
    x = processed_image
    x = np.squeeze(x, 0)

    VGG_MEAN = [123.68, 116.78, 103.94]
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    x = x + means
    return x


def gram_matrix(input_tensor):

    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def run(content_path, style_path, itrNum, content_weight, style_weight):

    # get the images
    tf.enable_eager_execution()

    content_image = load_img_preprocess(content_path)

    style_image = load_img_preprocess(style_path)

    """
    get the model from keras basically lets us extract the layers
    and their corresponding intermediate and batch outputs

    can do interesting things with the intermediate layers results

    """

    pretrained_net = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    pretrained_net.trainable = False

    style_layers = []

    for i in range(1, 5):
        style_layers.append('block{0}_conv1'.format(i))

    content_out_layers = [pretrained_net.get_layer('block5_conv2')]

    style_out_layers = []

    for layer_name in style_layers:
        style_out_layers.append(pretrained_net.get_layer(layer_name))

    model_out = content_out_layers + style_out_layers

    main_model = models.Model(pretrained_net.input, model_out)

    for layer in main_model.layers:
        layer.trainable = False

    style_features = [style_layer[0]
                      for style_layer in main_model(style_image)[:5]]
    content_features = [content_layer[0]
                        for content_layer in main_model(content_image)[5:]]

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_img_preprocess(content_path)
    init_image = tensor_eager.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

















