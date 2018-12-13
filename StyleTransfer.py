from gooey import Gooey, GooeyParser
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfeager
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.python.keras.preprocessing import image as kp_image
import tensorflow.contrib.eager as tensor_eager
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

# default arguments
LEARNING_RATE = 1e1
BETA1 = 0.99
BETA2 = 0.999
EPSILON = 1e-1
STYLE_SCALE = 1.0
ITERATIONS = 5
POOLING = 'max'
OPTIMIZER = 'adam'

tf.enable_eager_execution()

@Gooey(advanced = True,
program_name = "Neural Style Transfer",
program_description = "Enter arguments in order to run the Style Transfer program.",
show_stop_warning = True,
force_stop_is_error = True,
show_success_modal = True,
run_validators = True,
show_sidebar = False,
image_dir = 'gooey_image_dir/',
progress_regex = r"^iteration: (?P<current>\d+)/(?P<total>\d+)$",
progress_expr="current/total * 100")
def build_parser():
    parser = GooeyParser(description="Choose input parameters for Neural Style Transfer")
    parser.add_argument('content', help='content image',
            metavar='CONTENT', widget = "FileChooser")
    parser.add_argument('style', help='style image',
            metavar='STYLE', widget = "FileChooser")
    parser.add_argument('--style2', help='additional style image',
            metavar='STYLE1', widget = "FileChooser")
    parser.add_argument('--style3', help='additional style image',
            metavar='STYLE2', widget = "FileChooser")
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='number of iterations to run for',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--optimizer',
            dest='optimizer', help='choose between Adam and Adagrad',
            metavar='OPTIMIZER', default=OPTIMIZER, choices = ['adam', 'adagrad'])
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter',
            metavar='EPSILON', default=EPSILON)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    content_image = load_img_preprocess(args.content)
    if(args.style2 is None):
        style_paths = [args.style]
    elif(args.style3 is None):
        style_paths=[args.style, args.style2]
    else:
        style_paths=[args.style, args.style2, args.style3]
    style_images = []

    #make an array and add all style images produced from style paths given as input
    for i in style_paths:
    #call the load and preprocess function to load the style images from the given path
        print(i)
        style_images.append(load_img_preprocess(i))
    """
    getting the model from keras  lets us extract the layers
    and their corresponding intermediate and batch outputs
    """

    pretrained_net = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    pretrained_net.trainable = False

    style_layers = []

    for i in range(1, 5):
        style_layers.append('block{}_conv1'.format(i))

    content_out_layers = [pretrained_net.get_layer('block5_conv2').output]

    style_out_layers = []

    for layer_name in style_layers:
        style_out_layers.append(pretrained_net.get_layer(layer_name).output)

    model_out = content_out_layers + style_out_layers

    main_model = models.Model(pretrained_net.input, model_out)

    for layer in main_model.layers:
        layer.trainable = False
    style_features = []
    for i in style_images:
        style_features.append([style_layer[0]
                        for style_layer in main_model(i)[:5]])
        
    content_features = [content_layer[0]
                        for content_layer in main_model(content_image)[5:]]
    gram_style_features = []

    for i in style_features:
        gram_style_features.append([gram_matrix(style_feature) for style_feature in i])

    # Set initial image
    init_image = load_img_preprocess(args.content)
    init_image = tensor_eager.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    if args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    else:
        opt = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)

    loss_best = float('inf')
    best_img = None

    # clip the input image 
    # using max dim and min dim
    means = np.array([103.939, 116.779, 123.68])
    min_dim = -1 * means
    max_dim = 255-means

    for i in range(args.iterations):
        # compute loss for all layers
        # compute style loss 
        # compute content loss
        # compute their sum and produce gradients over the total loss
        # optimize using the total loss and the input image
        print("iteration: {}/{}".format(i+1, args.iterations))
        with tf.GradientTape() as gradi:
            out_final_entireImage = main_model(init_image)
            style_out = out_final_entireImage[:5]
            content_out = out_final_entireImage[5:]
            stylePoints = 0
            contentPoints = 0

            # equal weight across the contributions of all layers
            content_layer_norm = 1/float(5)
            for out, inter in zip(content_features, content_out):
                contentPoints= contentPoints + content_layer_norm*(tf.reduce_mean(tf.square(inter[0] - out)))

            
            style_layer_norm = 1/float(5)
            for grams in gram_style_features:
                for out, inter in zip(grams, style_out):
                    stylePoints = stylePoints + style_layer_norm*get_style_loss(inter[0],out)

            contentPoints = contentPoints*contentPoints
            stylePoints = stylePoints*stylePoints

            totalLoss = contentPoints + stylePoints

        gradients = gradi.gradient(totalLoss, init_image)
        opt.apply_gradients([(gradients,init_image)])

        clip_initImage = tf.clip_by_value(init_image, min_dim, max_dim)

        init_image.assign(clip_initImage)

        if(totalLoss<loss_best):
            loss_best = totalLoss
            best_img = deprocess_img(init_image.numpy())
        
    plt.figure(figsize=(10,10))
    display_img = np.squeeze(best_img, axis=0)
    plt.imshow(display_img)
    plt.title('output image')
    itr = str(args.iterations)
    output_name = "result"+itr+"itr"+"-"+datetime.now().strftime("%m%d-%H%M")+".jpg"
    plt.imsave(output_name, display_img)
    plt.show()

# referece
# citing URL: https://keras.io/applications/#vgg19
def load_img_preprocess(image_path):
    img_str = tf.read_file(image_path)
    img_decode = tf.image.decode_jpeg(img_str, 3)
    img = tf.cast(img_decode, tf.float32)
    dim =512.0
    height = tf.to_float(tf.shape(img)[1])
    width = tf.to_float(tf.shape(img)[0])
    print('this is the old height and width ', height, width)
    scale = tf.cond(tf.greater(height, width), lambda: dim/width , lambda: dim/height)
    print('this is the scale ', scale)
    newHeight = tf.to_int32(height * scale)
    newWidth = tf.to_int32(width * scale)
    print('newheight and new width', newHeight, newWidth)
    img = tf.image.resize_images(img, [newHeight, newWidth])
    img = np.expand_dims(img, axis=0)
    VGG_MEAN = [123.68, 116.78, 103.94]
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    img = img - means
    max_dim = 512
    img = Image.open(image_path)
    norm = max(img.size)
    scale = max_dim/norm
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img
    # performing the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def restore_image(processed_image):
    x = processed_image
    x = np.squeeze(x, 0)
    VGG_MEAN = [123.68, 116.78, 103.94]
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    x = x + means
    return x

def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

if __name__ == '__main__':
    main()