"""Code for training SIFA."""
from datetime import datetime
#import json
import numpy as np
import random
import os
#import cv2
import time
import argparse
import h5py
import logging
import platform
import skimage.io

import data_loader
import losses_prostate as losses
import model_prostate as model
from stats_func import *
import threading
import pdb
import SimpleITK as sitk
import sys

if platform.system() == "Windows":
    sys.path.append(r"D:\Dropbox\sourcecode\python\util")
else:
    sys.path.append("/home/chenxu31/sourcecode/python/util")
import common_net
import common_prostate
import common_metrics

logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel(logging.DEBUG)

if sys.version_info.major < 3:
    import Queue
else:
    import queue as Queue

random_seed = 1234


def _label_decomp(num_cls, label_vol):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    _batch_shape = label_vol.shape
    _vol = np.zeros(_batch_shape)
    _vol[label_vol == 0] = 1
    _vol = _vol[..., np.newaxis]
    for i in range(num_cls):
        if i == 0:
            continue
        _n_slice = np.zeros(label_vol.shape)
        _n_slice[label_vol == i] = 1
        _vol = np.concatenate( (_vol, _n_slice[..., np.newaxis]), axis = 3 )
    return np.float32(_vol[:, :, :, :, 0])


class SIFA:
    """The SIFA module."""
    def __init__(self, args):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._data_dir = args.data_dir
        self._output_root_dir = args.output_root_dir
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(args.pool_size)
        self._lambda_a = float(args._LAMBDA_A)
        self._lambda_b = float(args._LAMBDA_B)
        self._skip = bool(args.skip)
        self._num_cls = int(args.num_cls)
        self._base_lr = float(args.base_lr)
        self._max_step = int(args.max_step)
        self._keep_rate_value = float(args.keep_rate_value)
        self._is_training_value = bool(args.is_training_value)
        self._batch_size = int(args.batch_size)
        self._lr_gan_decay = bool(args.lr_gan_decay)
        self._to_restore = bool(args.to_restore)
        self._checkpoint_dir = args.checkpoint_dir
        self._save_interval = args.save_interval
        self._evaluation_interval = args.evaluation_interval

        self.data_iter = common_prostate.DataIter(args.data_dir, patch_depth=args.patch_depth, batch_size=args.batch_size)

        self.fake_images_A = np.zeros(
            (self._pool_size, self._batch_size, self.data_iter.patch_height, self.data_iter.patch_width, 1))
        self.fake_images_B = np.zeros(
            (self._pool_size, self._batch_size, self.data_iter.patch_height, self.data_iter.patch_width, 1))

        _, self.val_data_t, _, self.val_label_t = common_prostate.load_val_data(args.data_dir)
        self.test_data_s, self.test_data_t, self.test_label_s, self.test_label_t = common_prostate.load_test_data(args.data_dir)
        self.val_data_t_shape = self.val_data_t.shape[1:]
        self.test_data_s_shape = self.test_data_s.shape[1:]
        self.test_data_t_shape = self.test_data_t.shape[1:]

    def model_setup(self):
        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                self.data_iter.patch_height,
                self.data_iter.patch_width,
                1
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                self.data_iter.patch_height,
                self.data_iter.patch_width,
                1
            ], name="input_B")
        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                self.data_iter.patch_height,
                self.data_iter.patch_width,
                1
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                self.data_iter.patch_height,
                self.data_iter.patch_width,
                1
            ], name="fake_pool_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                None,
                self.data_iter.patch_height,
                self.data_iter.patch_width,
                self._num_cls
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                None,
                self.data_iter.patch_height,
                self.data_iter.patch_width,
                self._num_cls
            ], name="gt_B") # for validation only, not used during training

        self.keep_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.num_fake_inputs = 0

        self.learning_rate_gan = tf.placeholder(tf.float32, shape=[], name="lr_gan")
        self.learning_rate_seg = tf.placeholder(tf.float32, shape=[], name="lr_seg")

        self.lr_gan_summ = tf.summary.scalar("lr_gan", self.learning_rate_gan)
        self.lr_seg_summ = tf.summary.scalar("lr_seg", self.learning_rate_seg)

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(inputs, self.data_iter.get_shapes(), self._batch_size, self._num_cls,
                                    skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.pred_mask_a = outputs['pred_mask_a']
        self.pred_mask_b = outputs['pred_mask_b']
        self.pred_mask_b_ll = outputs['pred_mask_b_ll']
        self.pred_mask_fake_a = outputs['pred_mask_fake_a']
        self.pred_mask_fake_b = outputs['pred_mask_fake_b']
        self.pred_mask_fake_b_ll = outputs['pred_mask_fake_b_ll']
        self.prob_pred_mask_fake_b_is_real = outputs['prob_pred_mask_fake_b_is_real']
        self.prob_pred_mask_b_is_real = outputs['prob_pred_mask_b_is_real']
        self.prob_pred_mask_fake_b_ll_is_real = outputs['prob_pred_mask_fake_b_ll_is_real']
        self.prob_pred_mask_b_ll_is_real = outputs['prob_pred_mask_b_ll_is_real']

        self.prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
        self.prob_fake_pool_a_aux_is_real = outputs['prob_fake_pool_a_aux_is_real']
        self.prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']
        
        self.predicter_fake_b = tf.nn.softmax(self.pred_mask_fake_b)
        self.compact_pred_fake_b = tf.argmax(self.predicter_fake_b, 3)
        self.compact_y_fake_b = tf.argmax(self.gt_a, 3)

        self.predicter_b = tf.nn.softmax(self.pred_mask_b)
        self.compact_pred_b = tf.argmax(self.predicter_b, 3)
        self.compact_y_b = tf.argmax(self.gt_b, 3)

        self.dice_fake_b_arr = dice_eval(self.compact_pred_fake_b, self.gt_a, self._num_cls)
        self.dice_fake_b_mean = tf.reduce_mean(self.dice_fake_b_arr)
        self.dice_fake_b_mean_summ = tf.summary.scalar("dice_fake_b", self.dice_fake_b_mean)

        self.dice_b_arr = dice_eval(self.compact_pred_b, self.gt_b, self._num_cls)
        self.dice_b_mean = tf.reduce_mean(self.dice_b_arr)
        self.dice_b_mean_summ = tf.summary.scalar("dice_b", self.dice_b_mean)

    def compute_losses(self):

        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)
        lsgan_loss_p = losses.lsgan_loss_generator(self.prob_pred_mask_b_is_real)
        lsgan_loss_p_ll = losses.lsgan_loss_generator(self.prob_pred_mask_b_ll_is_real)
        lsgan_loss_a_aux = losses.lsgan_loss_generator(self.prob_fake_a_aux_is_real)

        ce_loss_b, dice_loss_b = losses.task_loss(self.pred_mask_fake_b, self.gt_a, self._num_cls)
        ce_loss_b_ll, dice_loss_b_ll = losses.task_loss(self.pred_mask_fake_b_ll, self.gt_a, self._num_cls)
        l2_loss_b = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/s_B/' in v.name or '/s_B_ll/' in v.name or '/e_B/' in v.name])


        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        seg_loss_B = ce_loss_b + dice_loss_b + l2_loss_b + 0.1 * (ce_loss_b_ll + dice_loss_b_ll) + 0.1 * g_loss_B + 0.1 * lsgan_loss_p + 0.01 * lsgan_loss_p_ll + 0.1 * lsgan_loss_a_aux

        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_A_aux = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_cycle_a_aux_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_aux_is_real,
        )
        d_loss_A = d_loss_A + d_loss_A_aux
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )
        d_loss_P = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_pred_mask_fake_b_is_real,
            prob_fake_is_real=self.prob_pred_mask_b_is_real,
        )
        d_loss_P_ll = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_pred_mask_fake_b_ll_is_real,
            prob_fake_is_real=self.prob_pred_mask_b_ll_is_real,
        )

        optimizer_gan = tf.train.AdamOptimizer(self.learning_rate_gan, beta1=0.5)
        optimizer_seg = tf.train.AdamOptimizer(self.learning_rate_seg)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if '/d_A/' in var.name]
        d_B_vars = [var for var in self.model_vars if '/d_B/' in var.name]
        g_A_vars = [var for var in self.model_vars if '/g_A/' in var.name]
        e_B_vars = [var for var in self.model_vars if '/e_B/' in var.name]
        de_B_vars = [var for var in self.model_vars if '/de_B/' in var.name]
        s_B_vars = [var for var in self.model_vars if '/s_B/' in var.name]
        s_B_ll_vars = [var for var in self.model_vars if '/s_B_ll/' in var.name]
        d_P_vars = [var for var in self.model_vars if '/d_P/' in var.name]
        d_P_ll_vars = [var for var in self.model_vars if '/d_P_ll/' in var.name]

        self.d_A_trainer = optimizer_gan.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer_gan.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer_gan.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer_gan.minimize(g_loss_B, var_list=de_B_vars)
        self.d_P_trainer = optimizer_gan.minimize(d_loss_P, var_list=d_P_vars)
        self.d_P_ll_trainer = optimizer_gan.minimize(d_loss_P_ll, var_list=d_P_ll_vars)
        self.s_B_trainer = optimizer_seg.minimize(seg_loss_B, var_list=e_B_vars + s_B_vars + s_B_ll_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
        self.ce_B_loss_summ = tf.summary.scalar("ce_B_loss", ce_loss_b)
        self.dice_B_loss_summ = tf.summary.scalar("dice_B_loss", dice_loss_b)
        self.l2_B_loss_summ = tf.summary.scalar("l2_B_loss", l2_loss_b)
        self.s_B_loss_summ = tf.summary.scalar("s_B_loss", seg_loss_B)
        self.s_B_loss_merge_summ = tf.summary.merge([self.ce_B_loss_summ, self.dice_B_loss_summ, self.l2_B_loss_summ, self.s_B_loss_summ])
        self.d_P_loss_summ = tf.summary.scalar("d_P_loss", d_loss_P)
        self.d_P_ll_loss_summ = tf.summary.scalar("d_P_loss_ll", d_loss_P_ll)
        self.d_P_loss_merge_summ = tf.summary.merge([self.d_P_loss_summ, self.d_P_ll_loss_summ])

    def save_images(self, sess, step):

        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['inputA_', 'inputB_', 'fakeA_',
                 'fakeB_', 'cycA_', 'cycB_']

        with open(os.path.join(self._output_dir, 'step_' + str(step) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):

                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                inputs = {
                    'images_i': images_i,
                    'images_j': images_j,
                    'gts_i': gts_i,
                    'gts_j': gts_j,
                }

                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j'],
                    self.is_training:self._is_training_value,
                    self.keep_rate: self._keep_rate_value,
                })


                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(step) + "_" + str(i) + ".jpg"
                    skimage.io.imsave(os.path.join(self._images_dir, image_name), ((tensor[0] + 1) * 127.5).astype(np.uint8).squeeze())
                    v_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                v_html.write("<br>")

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=40)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            # Training Loop
            curr_lr_seg = self._base_lr#0.001
            cnt = -1

            best_dsc = 0
            best_test_dsc = 0
            best_test_assd = 0
            for i in range(self._max_step):
                #starttime = time.time()

                cnt += 1
                curr_lr = self._base_lr

                patch_s, patch_t, label_s = self.data_iter.next()

                inputs = {
                    'images_i': patch_s,
                    'images_j': patch_t,
                    'gts_i': _label_decomp(self._num_cls, label_s),
                    'gts_j': _label_decomp(self._num_cls, label_s),
                }

                # Optimizing the G_A network
                _, fake_B_temp, summary_str = sess.run(
                    [self.g_A_trainer,
                     self.fake_images_b,
                     self.g_A_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.gt_a:
                            inputs['gts_i'],
                        self.learning_rate_gan: curr_lr,
                        self.keep_rate:self._keep_rate_value,
                        self.is_training:self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)

                fake_B_temp1 = self.fake_image_pool(
                    self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                # Optimizing the D_B network
                _, summary_str = sess.run(
                    [self.d_B_trainer, self.d_B_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.fake_pool_B: fake_B_temp1,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)

                # Optimizing the S_B network
                _, summary_str = sess.run(
                    [self.s_B_trainer, self.s_B_loss_merge_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.gt_a:
                            inputs['gts_i'],
                        self.learning_rate_seg: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }

                )
                writer.add_summary(summary_str, cnt)

                # Optimizing the G_B network
                _, fake_A_temp, summary_str = sess.run(
                    [self.g_B_trainer,
                     self.fake_images_a,
                     self.g_B_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.gt_a: inputs['gts_i'],
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)

                fake_A_temp1 = self.fake_image_pool(
                    self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                # Optimizing the D_A network
                _, summary_str = sess.run(
                    [self.d_A_trainer, self.d_A_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.fake_pool_A: fake_A_temp1,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)

                # Optimizing the D_P network
                _, summary_str = sess.run(
                    [self.d_P_trainer, self.d_P_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)

                # Optimizing the D_P_ll network
                _, summary_str = sess.run(
                    [self.d_P_ll_trainer, self.d_P_ll_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)

                summary_str_gan, summary_str_seg = sess.run([self.lr_gan_summ, self.lr_seg_summ],
                         feed_dict={
                             self.learning_rate_gan: curr_lr,
                             self.learning_rate_seg: curr_lr_seg,
                         })

                writer.add_summary(summary_str_gan, cnt)
                writer.add_summary(summary_str_seg, cnt)

                writer.flush()
                self.num_fake_inputs += 1

                #print ('iter {}: processing time {}'.format(cnt, time.time() - starttime))
                
                # batch evaluation
                if (i + 1) % self._evaluation_interval == 0:
                    summary_str_fake_b, summary_str_b = sess.run([self.dice_fake_b_mean_summ, self.dice_b_mean_summ],
                                                                 feed_dict={
                                                                     self.input_a: inputs['images_i'],
                                                                     self.gt_a: inputs['gts_i'],
                                                                     self.input_b: inputs['images_j'],
                                                                     self.gt_b: inputs['gts_j'],
                                                                     self.is_training: False,
                                                                     self.keep_rate: 1.0,
                                                                 })
                    writer.add_summary(summary_str_fake_b, cnt)
                    writer.add_summary(summary_str_b, cnt)
                    writer.flush()

                    val_t_dsc, test_t_dsc, test_t_assd, test_t_pred_list = self.validate(sess)
                    logging.info("Validation (%d) val_dsc:%f/%f  test_dsc:%f/%f  test_assd:%f/%f"%
                                 (i + 1, val_t_dsc.mean(), val_t_dsc.std(), test_t_dsc.mean(), test_t_dsc.std(),
                                  test_t_assd.mean(), test_t_assd.std()))

                    if val_t_dsc.mean() > best_dsc:
                        best_dsc = val_t_dsc.mean()
                        best_test_dsc = test_t_dsc
                        best_test_assd = test_t_assd
                        saver.save(sess, os.path.join(self._output_dir, "sifa_best"), global_step=0)


                """
                if (cnt+1) % self._save_interval ==0:

                    self.save_images(sess, cnt)
                    saver.save(sess, os.path.join(
                        self._output_dir, "sifa"), global_step=cnt)
              """
            saver.save(sess, os.path.join(self._output_dir, "sifa"), global_step=cnt)

            logging.info("Final val_best_dsc:%f  test_best_dsc:%f/%f  test_best_assd:%f/%f" %
                         (best_dsc, best_test_dsc.mean(), best_test_dsc.std(), best_test_assd.mean(), best_test_assd.std()))

            writer.add_graph(sess.graph)

    def validate(self, sess):
        patch_shape = (self.data_iter.patch_height, self.data_iter.patch_width, self.data_iter.patch_depth)

        common_feed = {
            self.is_training: False,
            self.keep_rate: 1.0,
        }

        test_t_pred_list = []
        val_t_dsc_list = np.zeros((self.val_data_t.shape[0], self._num_cls - 1), np.float32)
        test_t_dsc_list = np.zeros((self.test_data_t.shape[0], self._num_cls - 1), np.float32)
        test_t_assd_list = np.zeros((self.test_data_t.shape[0], self._num_cls - 1), np.float32)
        for i in range(self.val_data_t.shape[0]):
            t_pred = common_net.produce_results(sess, self.input_b, self.predicter_b, self.val_data_t[i], patch_shape,
                                                common_feed=common_feed, batch_size=self._batch_size,
                                                num_classes=self._num_cls, is_seg=True)
            t_pred = t_pred.argmax(-1).astype(np.float32)

            t_dsc = common_metrics.calc_multi_dice(t_pred, self.val_label_t[i], num_cls=self._num_cls)
            val_t_dsc_list[i] = t_dsc

        for i in range(self.test_data_t.shape[0]):
            t_pred = common_net.produce_results(sess, self.input_b, self.predicter_b, self.test_data_t[i], patch_shape,
                                                common_feed=common_feed, batch_size=self._batch_size,
                                                num_classes=self._num_cls, is_seg=True)
            t_pred = t_pred.argmax(-1).astype(np.float32)
            test_t_pred_list.append(t_pred)

            t_dsc = common_metrics.calc_multi_dice(t_pred, self.test_label_t[i], num_cls=self._num_cls)
            t_assd = common_metrics.calc_multi_assd(t_pred, self.test_label_t[i], num_cls=self._num_cls)
            test_t_dsc_list[i] = t_dsc
            test_t_assd_list[i] = t_assd

        return val_t_dsc_list, test_t_dsc_list, test_t_assd_list, test_t_pred_list

def main(args):
    
    tf.set_random_seed(random_seed)

    sifa_model = SIFA(args)
    sifa_model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/shenlab/lab_stor/xuchen/data/prostate/h5_data', help='path of the dataset')
    parser.add_argument('--output_root_dir', type=str, default=r'/shenlab/lab_stor/xuchen/logs/sifa/temp', help='output path')
    parser.add_argument('--checkpoint_dir', type=str, default=r'/shenlab/lab_stor/xuchen/checkpoints/sifa/temp', help='checkpoint path')
    parser.add_argument('--patch_depth', type=int, default=1, help="patch depth")
    parser.add_argument('--pool_size', type=int, default=50, help="pool size ")
    parser.add_argument('--_LAMBDA_A', type=int, default=10, help="")
    parser.add_argument('--_LAMBDA_B', type=int, default=10, help="")
    parser.add_argument('--skip', type=int, default=1, help="")
    parser.add_argument('--num_cls', type=int, default=4, help="")
    parser.add_argument('--base_lr', type=float, default=0.0001, help="")
    parser.add_argument('--max_step', type=int, default=20000, help="")
    parser.add_argument('--keep_rate_value', type=float, default=1.0, help="")
    parser.add_argument('--is_training_value', type=int, default=1, help="")
    parser.add_argument('--batch_size', type=int, default=4, help="")
    parser.add_argument('--lr_gan_decay', type=float, default=0, help="")
    parser.add_argument('--to_restore', type=int, default=0, help="")
    parser.add_argument('--save_interval', type=int, default=20000, help="")
    parser.add_argument('--evaluation_interval', type=int, default=1000, help="")

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    main(args)
