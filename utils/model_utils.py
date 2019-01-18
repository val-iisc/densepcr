import tensorflow as tf
import tflearn
from utils.encoders_decoders import *
import utils.pointnet2_utils.tf_util
from utils.pointnet2_utils.pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg


### conv --> bn --> relu
def _conv2d(layer,num_filters,filter_size,strides,activation='relu',weight_decay=1e-5,regularizer='L2',bn=False):
	layer = tflearn.layers.conv.conv_2d(layer,num_filters,filter_size,strides=strides,activation='linear',weight_decay=weight_decay,regularizer=regularizer)
	if bn:
		layer = tflearn.layers.normalization.batch_normalization(layer)
	if activation == 'relu':
		layer = tflearn.activations.relu(layer)
	return layer


def _deconv2d(layer,num_neurons,filter_size,output_shape,strides=2,activation='linear',weight_decay=1e-5,regularizer='L2'):
	layer = tflearn.layers.conv.conv_2d_transpose(layer,num_neurons,filter_size,output_shape,strides=strides,activation=activation,weight_decay=weight_decay,regularizer=regularizer)
	return layer


### conv --> bn --> relu
def _fc(layer,num_neurons,activation='relu',weight_decay=1e-3,regularizer='L2',bn=False):
	layer = tflearn.layers.core.fully_connected(layer,num_neurons,activation='linear',weight_decay=weight_decay,regularizer=regularizer)
	if bn:
		layer = tflearn.layers.normalization.batch_normalization(layer)
	if activation=='relu':
		layer = tflearn.activations.relu(layer)
	return layer	

###
def basenet(img_inp):
	x=img_inp
	#128 128
	x=_conv2d(x,32,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,32,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,64,(3,3),2,bn=FLAGS.bn)
	#64 64
	x=_conv2d(x,64,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,64,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,128,(3,3),2,bn=FLAGS.bn)
	#32 32
	x=_conv2d(x,128,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,128,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,256,(3,3),2,bn=FLAGS.bn)
	#16 16
	x=_conv2d(x,256,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,256,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,512,(3,3),2,bn=FLAGS.bn)
	#8 8
	x=_conv2d(x,512,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,512,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,512,(3,3),1,bn=FLAGS.bn)
	x=_conv2d(x,512,(5,5),2,bn=FLAGS.bn)
	#4 4
	x=_fc(x,FLAGS.bottleneck,bn=FLAGS.bn)
	x=_fc(x,256,bn=FLAGS.bn)
	x=_fc(x,256,bn=FLAGS.bn)
	x=_fc(x,PCL_SIZE*3,activation='linear',bn=False)
	x=tf.reshape(x,(-1,PCL_SIZE,3))
	return x


# Dense Network
def densenet(pcl):
	print 'Input resolution: %d; Output resolution: %d'%(IN_PCL_SIZE, OUT_PCL_SIZE)
	with tf.variable_scope('densenet_%dto%d'%(IN_PCL_SIZE,OUT_PCL_SIZE)) as scope:
		z = encoder_with_convs_and_symmetry(in_signal=pcl, n_filters=[32,64,64], 
			filter_sizes=[1],
			strides=[1],
			b_norm=FLAGS.bn_encoder,
			verbose=True,
			scope=scope
			)
		print 'z: ', z.shape
		print 'pcl_in: ', pcl.shape
		point_feat = tf.tile(tf.expand_dims(pcl, 2), [1,1,UPSAMPLING_FACTOR,1]) # (bs,NUM_POINTS,3) --> (bs,NUM_POINTS,1,3) --> (bs,NUM_POINTS,4,3)
		point_feat = tf.reshape(point_feat, [BATCH_SIZE, OUT_PCL_SIZE, 3]) # (bs,NUM_POINTS,4,3) --> (bs,NUM_UPSAMPLE_POINTS,3)
		print 'point_feat: ', point_feat.shape
		global_feat = tf.expand_dims(z, axis=1) # (bs,bneck) --> (bs,1,bneck)
		print 'global_feat: ', global_feat.shape
		global_feat = tf.tile(global_feat, [1, OUT_PCL_SIZE, 1]) # (bs,1,bneck) --> (bs,NUM_UPSAMPLE_POINTS,bneck)
		print 'global_feat: ', global_feat.shape
		concat_feat = tf.concat([point_feat, global_feat], axis=2)
		print 'concat_feat: ', concat_feat.shape
		if FLAGS.fold:
			grid = tf.meshgrid(tf.linspace(-0.1,0.1,GRID_SIZE), tf.linspace(-0.1,0.1,GRID_SIZE))
			grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
			print 'grid: ', grid.shape
			grid_feat = tf.tile(grid, [BATCH_SIZE, IN_PCL_SIZE, 1])
			print 'grid_feat: ', grid_feat.shape
			concat_feat = tf.concat([concat_feat, grid_feat], axis=2)
			print 'concat_feat: ', concat_feat.shape
		if FLAGS.local:
			_, local_feat, _ = pointnet_sa_module(pcl, None, npoint=IN_PCL_SIZE, radius=radius[IN_PCL_SIZE], nsample=8, mlp=[32,32,64], mlp2=None, group_all=False, is_training=False, bn_decay=None, scope='local_feat', bn=False)
			print 'local_feat: ', local_feat.shape
			local_feat = tf.tile(tf.expand_dims(local_feat, 2), [1,1,UPSAMPLING_FACTOR,1]) # (bs,NUM_POINTS,64) --> (bs,NUM_POINTS,1,64) --> (bs,NUM_POINTS,4,64)
			local_feat = tf.reshape(local_feat, [BATCH_SIZE, OUT_PCL_SIZE, -1]) # (bs,NUM_POINTS,4,64) --> (bs,NUM_UPSAMPLE_POINTS,64)
			concat_feat = tf.concat([concat_feat, local_feat], axis=2)
			print 'concat_feat: ', concat_feat.shape
		elif FLAGS.multiple_local:
			radius_list = [round(radius[IN_PCL_SIZE]*scale,2) for scale in [1.,1.1,1.2]]
			_, local_feat = pointnet_sa_module_msg(pcl, None, npoint=IN_PCL_SIZE, radius_list=radius_list, nsample_list=[8,16,32], mlp_list=[[8,16], [16,32], [16,32]], is_training=False, bn_decay=None, scope='multiple_local_feat', bn=False)
			print 'local_feat: ', local_feat.shape
			local_feat = tf.tile(tf.expand_dims(local_feat, 2), [1,1,UPSAMPLING_FACTOR,1]) # (bs,NUM_POINTS,64) --> (bs,NUM_POINTS,1,64) --> (bs,NUM_POINTS,4,64)
			local_feat = tf.reshape(local_feat, [BATCH_SIZE, OUT_PCL_SIZE, -1]) # (bs,NUM_POINTS,4,64) --> (bs,NUM_UPSAMPLE_POINTS,64)
			concat_feat = tf.concat([concat_feat, local_feat], axis=2)
			print 'concat_feat: ', concat_feat.shape
		
		dense = decoder_with_convs_only(concat_feat, n_filters=[64,128,128,3], 
			filter_sizes=[1], 
			strides=[1],
			b_norm=FLAGS.bn_decoder, 
			b_norm_finish=False, 
			verbose=True, 
			scope=scope)
		print 'dense: ', dense.shape
		return dense