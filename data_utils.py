import json
import os
import re
import sys
import tensorflow as tf
from itertools import product
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename

from utils.shapenet_taxonomy import shapenet_id_to_category, shapenet_category_to_id


NUM_VIEWS = 24
PNG_FILES = [(str(i).zfill(2)+'.png') for i in xrange(NUM_VIEWS)]


def create_folder(folder):
	if not exists(folder):
		makedirs(folder)


def scale(gt_pc, pr_pc): #pr->[-1,1], gt->[-1,1]

	pred = tf.cast(pr_pc, dtype=tf.float32)
	gt   = tf.cast(gt_pc, dtype=tf.float32)

	min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)])
	max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)])
	min_pr = tf.convert_to_tensor([tf.reduce_min(pred[:,:,i], axis=1) for i in xrange(3)])
	max_pr = tf.convert_to_tensor([tf.reduce_max(pred[:,:,i], axis=1) for i in xrange(3)])

	length_gt = tf.abs(max_gt - min_gt)
	length_pr = tf.abs(max_pr - min_pr)

	diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt
	diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr

	new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)])
	new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)])
	new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)])
	new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)])

	size_pr = tf.reduce_max(length_pr, axis=0)
	size_gt = tf.reduce_max(length_gt, axis=0)

	scaling_factor_gt = 2. / size_gt # 2. is the length of the [-1,1] cube
	scaling_factor_pr = 2. / size_pr

	box_min = tf.ones_like(new_min_gt) * -1.

	adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
	adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr

	pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
	gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))

	return pred_scaled, gt_scaled


def get_shapenet_models(FLAGS, NUM_VIEWS=None):
	'''
	Training and validation set creation
	Args:
		FLAGS: arguments parsed for the particular experiment
		NUM_VIEWS: Number of views rendered for every model in the training set
	Returns:
		train_models: list of models (absolute path to each model) in the training set
		val_models:	list of models (absolute path to each model) in the validation set
		train_pair_indices: list of ind pairs for training set
		val_pair_indices: list of ind pairs for validation set
		-->	ind[0] : model index (range--> [0, len(models)-1])
		-->	ind[1] : view index (range--> [0, NUM_VIEWS-1])
	'''


	if FLAGS.category == 'all':
		cats = shapenet_id_to_category.keys()
	else:
		cats = [shapenet_category_to_id[FLAGS.category]]

	with open(join(BASE_DIR, 'split/train_models.json'), 'r') as f:
		train_models_dict = json.load(f)

	with open(join(BASE_DIR, 'split/val_models.json'), 'r') as f:
		val_models_dict = json.load(f)

	train_models = []
	val_models = []

	for cat in cats:
		train_models.extend([join(data_dir, model) for model in train_models_dict[cat]])

	for cat in cats:
		val_models.extend([join(data_dir, model) for model in val_models_dict[cat]])

	val_models = val_models[:len(val_models)//4]

	if NUM_VIEWS:
		train_pair_indices = list(product(xrange(len(train_models)), xrange(NUM_VIEWS)))
		val_pair_indices = list(product(xrange(len(val_models)), xrange(NUM_VIEWS)))
		print 'TRAINING: models={}  samples={}'.format(len(train_models),len(train_models)*NUM_VIEWS)
		print 'VALIDATION: models={}  samples={}'.format(len(val_models),len(val_models)*NUM_VIEWS)
		print
		return train_models, val_models, train_pair_indices, val_pair_indices

	else:
		print 'TRAINING: models={}'.format(len(train_models))
		print 'VALIDATION: models={}'.format(len(val_models))
		print
		return train_models, val_models
