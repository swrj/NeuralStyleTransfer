from gooey import Gooey, GooeyParser
from argparse import ArgumentParser

CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 100
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

@Gooey(advanced = True,
    program_name = "Style Transfer",
    program_descirption = "Enter arguments in order to run the Style Transfer program.",
    show_stop_warning = True,
    force_stop_is_error = True,
    show_success_modal = True,
    run_validators = True,
    navigation = "Sidebar",
    show_sidebar = True,
    )
def main(): #TODO: add poll_external_updates to use dynamic values
    parser = GooeyParser(description="Test")
    sub_parsers = parser.add_subparsers()
    req_args = sub_parsers.add_parser('Required arguments')
    req_args.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True, widget = "FileChooser")
    req_args.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True, widget = "FileChooser")
    req_args.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True, widget = "DirChooser")
    opt_args = sub_parsers.add_parser('Advanced arguments')
    opt_args.add_argument('--iterations', type=int,
            dest='iterations', help='number of iterations to run for',
            metavar='ITERATIONS', default=ITERATIONS)
    '''
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output',
            help='checkpoint output format, e.g. output_{:05}.jpg or '
                 'output_%%05d.jpg',
            metavar='OUTPUT', default=None)
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS', default=None)
    parser.add_argument('--progress-write', default=False, action='store_true',
            help="write iteration progess data to OUTPUT's dir",
            required=False)
    parser.add_argument('--progress-plot', default=False, action='store_true',
            help="plot iteration progess data to OUTPUT's dir",
            required=False)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    '''
    opt_args.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    opt_args.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend',
            help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) '
                 '(default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    opt_args.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    opt_args.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    opt_args.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp',
            help='style layer weight exponentional increase - '
                 'weight(layer<n+1>) = weight_exp*weight(layer<n>) '
                 '(default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    opt_args.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    opt_args.add_argument('--tv-weight', type=float,
            dest='tv_weight',
            help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    opt_args.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    opt_args.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    opt_args.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    opt_args.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    '''
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend',
            help='ratio of blending initial image with normalized noise '
                 '(if no initial image specified, content image is used) '
                 '(default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    '''
    opt_args.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors',
            help='style-only transfer (preserving colors) - if color transfer '
                 'is not needed',
            widget='CheckBox')
    opt_args.add_argument('--pooling',
            dest='pooling',
            help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING, choices = ['max', 'avg'])
    opt_args.add_argument('--overwrite', action='store_true', dest='overwrite',
            help='write file even if there is already a file with that name')
    args = parser.parse_args()
    print (args)
    return parser
if __name__ == "__main__":
    main()
