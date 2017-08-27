from tf_unet import image_gen
from tf_unet import image_util
from tf_unet import unet
import json
import argparse
import numpy as np
import os


parser = argparse.ArgumentParser()
#admin
parser.add_argument("--action",     type=str, required=True, help="what to do: choices [] ")
parser.add_argument("--input_dir",  type=str, required=True, help="path to folder containing images")
parser.add_argument("--output_dir", type=str, required=True, help="where to put output files")
parser.add_argument("--chkpt_dir",  type=str, help="path to folder containing images")

parser.add_argument("--image_type", default="png", type=str)
parser.add_argument("--write_graph", default=False, type=bool)
parser.add_argument("--img_chan", default=2, type=int, help="which channel of image to use to gen masks")
parser.add_argument("--img_color", default=None, type=int, help="TODO - color(s) to use to gen masks")

#network
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_epochs",  default=1, type=int)
parser.add_argument("--num_layers",  default=3, type=int)
parser.add_argument("--z_dim",       default=100, type=int, help='include or exclude')
parser.add_argument("--keep_prob",  default=0.9, type=float)
#optimizer
parser.add_argument("--mode", default='train', type=str)
parser.add_argument("--display_step", default=100, type=int)
parser.add_argument("--optimizer",  default="momentum", type=str)
parser.add_argument("--opt_momentum",  default=0.2, type=float)
parser.add_argument("--opt_learning_rate",  default=0.2, type=float)
parser.add_argument("--decay_rate",  default=0.95, type=float)
parser.add_argument("--training_iters",  default=2, type=int)


a = parser.parse_args()

def print_args():
    for k, v in a._get_kwargs():
        print(k, "=", v)
    pass

def expirement():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    print_args()

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    search_path = a.input_dir + "*." + a.image_type
    generator = image_util.SplitImageDataProvider(search_path,
                                                  color=a.img_color,
                                                  chan=a.img_chan)

    net = unet.Unet(channels=generator.channels,
                    n_class=generator.n_class,
                    layers=a.num_layers,
                    cost="cross_entropy",
                    #keep_prob=a.keep_prob,
                    features_root=a.z_dim)

    trainer = unet.Trainer(net,
                           batch_size=a.batch_size,
                           optimizer=a.optimizer,
                           opt_kwargs=dict(momentum=a.opt_momentum,
                                           learning_rate=a.opt_learning_rate,
                                           decay_rate=a.decay_rate))

    path = trainer.train(generator,
                         a.output_dir,
                         training_iters=a.training_iters,
                         epochs=a.max_epochs,
                         dropout=a.keep_prob,
                         display_step=a.display_step,
                         restore=False,
                         write_graph=a.write_graph)
    print(path)
    #pass


def test():
    print_args()
    search_path = a.input_dir + "*." + a.image_type
    print(search_path)
    generator = image_util.SplitImageDataProvider(search_path,
                                                  color=a.img_color,
                                                  chan=a.img_chan)

    x_test, y_test = generator(5)
    #net = unet.Unet(channels=generator.channels,
    #                n_class=generator.n_class,
    #                layers=a.num_layers,
    #                features_root=a.z_dim)

    sess = tf.Session()
    # First let's load meta graph and restore weights

    pth = os.path.join(a.chkpt_dir, 'model.cpkt.meta')
    saver = tf.train.import_meta_graph(pth)
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    prediction = net.predict(a.chkpt_dir, x_test)
    plot_prediction(x_test, y_test, prediction, save=a.output_dir)
    print("done")
    #pass

def test_custom_gen(ig):
    print(ig.channels)
    x_test, y_test = ig(1)
    print(x_test.shape, y_test.shape)
    #misc.imshow(y_test)
    print(np.unique(y_test, return_counts=True))
    print(y_test.dtype)
    pass


def main():

    act = a.action
    print(a)

    if act == 'test-split':
        pth = '/home/psavine/data/test/unet_test/*'
        ig = image_util.SplitImageDataProvider(pth, a_min=255, a_max=255, color=None)
        test_custom_gen(ig)

    elif act == 'test2':
        ig = image_gen.GrayScaleDataProvider(256, 256, cnt=20)
        test_custom_gen(ig)
    elif act == 'test3':
        ig = image_util.ImageDataProvider('/home/psavine/data/test/unet_test2/*')
        test_custom_gen(ig)
    elif act == 'train':
        expirement()
    elif act == 'test':
        test()
    else:
        print('invalid action')
    pass


main()