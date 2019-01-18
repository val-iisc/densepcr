import os
import math
import tensorflow as tf

def update_best(saver, sess, snapshot_folder, best_folder, current_loss, best_loss, best_folder):
	if (current_loss < best_loss):
		print 'Best model found. Saving in %s...'%best_folder
		saver.save(sess, join(snapshot_folder, 'best', 'best'))
		os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
		return current_loss
	return best_loss


def load_previous_checkpoint(snapshot_folder, saver, sess):
	start_epoch = 0

	if FLAGS.load_best:
		ckpt_path = join(FLAGS.exp, 'best', 'best')
		print ('loading ' + ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)
		return

	ckpt = tf.train.get_checkpoint_state(snapshot_folder)
	if ckpt is not None:
		ckpt_path = ckpt.model_checkpoint_path
		print ('loading '+ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)
		start_epoch = 1 + int(re.match('.*-(\d*)$', ckpt_path).group(1))
	return start_epoch


def scale_emd(emd,num_points):
	return 0.01*emd/float(num_points)
