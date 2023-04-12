"""Code for testing SIFA."""
import numpy as np
import os
import pdb
import SimpleITK as sitk
import h5py
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model_prostate as model
import platform
import sys


if platform.system() == "Windows":
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    sys.path.append("/home/chenxu31/sourcecode/python/util")
import common_net
import common_prostate_ck
import common_metrics


def main(args):
    _, test_data, _, test_label = common_prostate_ck.load_test_data(args.data_dir)

    patch_height, patch_width = test_data.shape[1:3]

    input_a = tf.placeholder(
        tf.float32, [
            None,
            patch_height,
            patch_width,
            1
        ], name="input_A")
    input_b = tf.placeholder(
        tf.float32, [
            None,
            patch_height,
            patch_width,
            1
        ], name="input_B")
    fake_pool_A = tf.placeholder(
        tf.float32, [
            None,
            patch_height,
            patch_width,
            1
        ], name="fake_pool_A")
    fake_pool_B = tf.placeholder(
        tf.float32, [
            None,
            patch_height,
            patch_width,
            1
        ], name="fake_pool_B")

    inputs = {
        'images_a': input_a,
        'images_b': input_b,
        'fake_pool_a': fake_pool_A,
        'fake_pool_b': fake_pool_B,
    }
    data_shapes = {
        "patch_height": patch_height,
        "patch_width": patch_width,
    }
    keep_rate = tf.placeholder(tf.float32, shape=())
    is_training = tf.placeholder(tf.bool, shape=())
    outputs = model.get_outputs(inputs, data_shapes, 16, args.num_classes,
                                skip=True, is_training=is_training, keep_rate=keep_rate)

    patch_shape = (patch_height, patch_width, 1)
    pred_mask_b = outputs['pred_mask_b']
    predicter_b = tf.nn.softmax(pred_mask_b)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.checkpoint_dir)

        pred_list = []
        dsc_list = np.zeros((test_data.shape[0], args.num_classes - 1), np.float32)
        assd_list = np.zeros((test_data.shape[0], args.num_classes - 1), np.float32)
        common_feed = {
            is_training: False,
            keep_rate: 1.0,
        }
        for i in range(test_data.shape[0]):
            pred = common_net.produce_results(sess, input_b, predicter_b, test_data[i], patch_shape,
                                              common_feed=common_feed, batch_size=16,
                                              num_classes=args.num_classes, is_seg=True)
            pred = pred.argmax(-1).astype(np.float32)
            pred_list.append(pred)

            dsc = common_metrics.calc_multi_dice(pred, test_label[i], args.num_classes)
            assd = common_metrics.calc_multi_assd(pred, test_label[i], args.num_classes)

            dsc_list[i, :] = dsc
            assd_list[i, :] = assd

    msg = "dsc:%f/%f  assd:%f/%f" % (dsc_list.mean(), dsc_list.std(), assd_list.mean(), assd_list.std())
    for c in range(args.num_classes - 1):
        name = common_prostate_ck.MAP_ID_NAME[c + 1]
        dsc = dsc_list[:, c]
        assd = assd_list[:, c]
        msg += "  %s_dsc:%f/%f  %s_assd:%f/%f" % (name, dsc.mean(), dsc.std(), name, assd.mean(), assd.std())

    print(msg)
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
            f.write(msg)

        np.save(os.path.join(args.output_dir, "dsc.npy"), dsc_list)
        np.save(os.path.join(args.output_dir, "assd.npy"), assd_list)

        for i,im in enumerate(pred_list):
            common_prostate_ck.save_nii(im, os.path.join(args.output_dir, "pred_%d.nii.gz" % (i + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'F:\datasets\pelvic\h5_data')
    parser.add_argument("--checkpoint_dir", type=str, default=r'E:\training\checkpoints\sifa\prostate\20210622-110814\sifa-99999')
    parser.add_argument("--output_dir", type=str, default=r'E:\training\test_output\sifa\pelvic_final')
    parser.add_argument("--num_classes", type=int, default=3) #TODOck
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    main(args)
