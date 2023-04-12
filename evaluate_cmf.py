"""Code for testing SIFA."""
import numpy as np
import os
import pdb
import SimpleITK as sitk
import h5py
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import common_metrics
from main_cmf import DataIterUnpaired
from main_cmf import generate_image
import model_cmf


def save_nii(img, output_file):
    output_img = sitk.GetImageFromArray(img)
    sitk.WriteImage(output_img, output_file)


def main(args):
    data_iter = DataIterUnpaired(args.data_dir, 1, args.view)
    data_shapes = data_iter.get_shapes()

    input_a = tf.placeholder(
        tf.float32, [
            None,
            data_shapes["patch_height"],
            data_shapes["patch_width"],
            1
        ], name="input_A")
    input_b = tf.placeholder(
        tf.float32, [
            None,
            data_shapes["patch_height"],
            data_shapes["patch_width"],
            1
        ], name="input_B")
    fake_pool_A = tf.placeholder(
        tf.float32, [
            None,
            data_shapes["patch_height"],
            data_shapes["patch_width"],
            1
        ], name="fake_pool_A")
    fake_pool_B = tf.placeholder(
        tf.float32, [
            None,
            data_shapes["patch_height"],
            data_shapes["patch_width"],
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
    outputs = model.get_outputs(inputs, data_shapes, 16, 2, skip=True, is_training=is_training, keep_rate=keep_rate)

    pred_mask_b = outputs['pred_mask_b']
    predicter_b = tf.nn.softmax(pred_mask_b)

    patch_shape = (data_iter.patch_height, data_iter.patch_width, data_iter.patch_depth)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        checkpoint_file = os.path.join(args.checkpoint_dir, "sifa_best-0")
        saver.restore(sess, checkpoint_file)

        mri_seg_list = []
        mri_dsc_list = []
        mri_assd_list = []
        common_feed = {
            is_training: False,
            keep_rate: 1.0,
        }
        for i in range(data_iter.data_mri.shape[0]):
            mri_seg = generate_image(sess, common_feed, input_b, predicter_b, data_iter.data_mri[i],
                                     patch_shape, batch_size=16, is_seg=True)
            mri_seg = mri_seg.argmax(-1).astype(np.float32)
            mri_seg_list.append(mri_seg)

            mri_dsc = common_metrics.dc(data_iter.data_seg[i], mri_seg)
            mri_assd = common_metrics.assd(data_iter.data_seg[i], mri_seg)

            mri_dsc_list.append(mri_dsc)
            mri_assd_list.append(mri_assd)

    mri_dsc_list = np.array(mri_dsc_list)
    mri_assd_list = np.array(mri_assd_list)

    print("dsc:%f/%f  assd:%f/%f" % (mri_dsc_list.mean(), mri_dsc_list.std(), mri_assd_list.mean(), mri_assd_list.std()))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    np.save(os.path.join(args.output_dir, "mri_dsc.npy"), mri_dsc_list)
    np.save(os.path.join(args.output_dir, "mri_assd.npy"), mri_assd_list)

    for i,im in enumerate(mri_seg_list):
        save_nii(im, os.path.join(args.output_dir, "mri_seg_%d.nii.gz" % (i + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'D:\datasets\mri_seg_disentangle\146_180_175')
    parser.add_argument("--checkpoint_dir", type=str, default=r'D:\training\logs\SIFA\axial_full\20200226-101116')
    parser.add_argument("--output_dir", type=str, default=r'D:\training\test_output_sifa\axial_full')
    parser.add_argument("--view", type=str, default='axial')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    main(args)
