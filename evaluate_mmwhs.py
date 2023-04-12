"""Code for testing SIFA."""
import numpy as np
import os
import pdb
import SimpleITK as sitk
import h5py
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from main_cmf import generate_image
import model
import platform
import sys

if platform.system() == "Windows":
    sys.path.append(r"D:\Dropbox\sourcecode\python\util")
else:
    sys.path.append("/home/chenxu31/sourcecode/python/util")

import common_metrics


MAP_ID_NAME = {
    0: "BG",
    1: "MYO",
    2: "LAC",
    3: "LVC",
    4: "AA"
}
LABEL_COLORS = (
    (255, 0, 0), # red
    (0, 255, 0), # green
    (0, 0, 255), # blue
    (255, 255, 0), # yellow
    (255, 0, 255),
    (0, 255, 255),
)


def save_nii(img, output_file):
    img = img.transpose((2, 0, 1))
    output_img = sitk.GetImageFromArray(img)
    sitk.WriteImage(output_img, output_file)


def main(args):
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    f = h5py.File(os.path.join(args.data_dir, "ct_test.h5"), "r")
    test_data, test_label = np.array(f["data"]), np.array(f["label"])
    test_data = test_data.transpose((0, 2, 3, 1))
    test_label = test_label.transpose((0, 2, 3, 1))
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

    keep_rate = tf.placeholder(tf.float32, shape=())
    is_training = tf.placeholder(tf.bool, shape=())
    outputs = model.get_outputs(inputs, skip=True, is_training=is_training, keep_rate=keep_rate)

    pred_mask_b = outputs['pred_mask_b']
    predicter_b = tf.nn.softmax(pred_mask_b)

    patch_shape = (patch_height, patch_width, 1)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, args.checkpoint_dir)

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
            pred = generate_image(sess, common_feed, input_b, predicter_b, test_data[i],
                                  patch_shape, batch_size=16, is_seg=True, num_classes=args.num_classes)
            pred = pred.argmax(-1).astype(np.float32)
            pred_list.append(pred)

            dsc = common_metrics.calc_multi_dice(pred, test_label[i], args.num_classes)
            assd = common_metrics.calc_multi_assd(pred, test_label[i], args.num_classes)

            dsc_list[i, :] = dsc
            assd_list[i, :] = assd

    msg = "dsc:%f/%f  assd:%f/%f" % (dsc_list.mean(), dsc_list.std(), assd_list.mean(), assd_list.std())
    for c in range(args.num_classes - 1):
        name = MAP_ID_NAME[c + 1]
        dsc = dsc_list[c]
        assd = assd_list[c]
        msg += "  %s_dsc:%f/%f  %s_assd:%f/%f" % (name, dsc.mean(), dsc.std(), name, assd.mean(), assd.std())

    print(msg)

    if args.output_dir:
        np.save(os.path.join(args.output_dir, "dsc.npy"), dsc_list)
        np.save(os.path.join(args.output_dir, "assd.npy"), assd_list)

        for i,im in enumerate(pred_list):
            save_nii(im, os.path.join(args.output_dir, "seg_%d.nii.gz" % (i + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'D:\datasets\mri_seg_disentangle\146_180_175')
    parser.add_argument("--checkpoint_dir", type=str, default=r'D:\training\logs\SIFA\axial_full\20200226-101116')
    parser.add_argument("--output_dir", type=str, default=r'D:\training\test_output_sifa\mmwhs')
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    main(args)
