'''
Code for training the dense upsampling network, which consists of upsampling N points to U*N points, U being the upsampling factor. U=4 is used.
Loss used is Chamfer distance.
Three variants are tried:
	globlal_feature + grid_conditioning
	globlal_feature + local_feature + grid_conditioning
	globlal_feature + multiple_local_features + grid_conditioning
Run as:
	python train_upsampling.py --exp 1_fg --ip 132 --gpu 0 --category all --in_pcl_size 1024 --fold
	python train_upsampling.py --exp 2_fgl --ip 132 --gpu 0 --category all --in_pcl_size 1024  --fold --local
	python train_upsampling.py --exp 3_fgml --ip 132 --gpu 1 --category all --in_pcl_size 1024  --fold --multiple_local
'''

from importer import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment Prefixed with index')
parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--category', type=str, required=True, 
	help='Category to visualize from : \
	["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", "monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--in_pcl_size', type=int, required=True, 
	help='Size of input point cloud')
parser.add_argument('--fold', action='store_true', 
	help='Supply this parameter if you want to use local grid folding, otherwise ignore')
parser.add_argument('--local', action='store_true', 
	help='Supply this parameter if you want to use local point feature, otherwise ignore')
parser.add_argument('--multiple_local', action='store_true', 
	help='Supply this parameter if you want to use multiple local point features, otherwise ignore')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--bn_encoder', action='store_true', 
	help='Supply this parameter if you want batchnorm in the encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', 
	help='Supply this parameter if you want batchnorm in the decoder, otherwise ignore')
parser.add_argument('--lr', type=float, default=0.00005, 
	help='Learning Rate') ###
parser.add_argument('--print_n', type=int, default=100, 
	help='print output to terminal every n iterations')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

VAL_BATCH_SIZE = FLAGS.batch_size
BATCH_SIZE = FLAGS.batch_size
IN_PCL_SIZE = FLAGS.in_pcl_size

exp_dir = join('expts','upsampling',FLAGS.exp)


def fetch_batch(models, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	for ind in range(batch_num*batch_size, batch_num*batch_size+batch_size):
		model_path = models[ind]
		if IN_PCL_SIZE == 1024:
			pcl_gt = np.load(join(FLAGS.data_dir_pcl, model_path, pcl_4k_fname))
			pcl_ip = pcl_gt[:1024]
		elif IN_PCL_SIZE == 4096:
			pcl_gt = np.load(join(FLAGS.data_dir_pcl, model_path, pcl_16k_fname))
			pcl_ip = np.load(join(FLAGS.data_dir_pcl, model_path, pcl_4k_fname))
		batch_gt.append(pcl_gt)
		batch_ip.append(pcl_ip)
	batch_gt = np.array(batch_gt)
	batch_ip = np.array(batch_ip)
	return batch_ip, batch_gt


def get_epoch_loss(models):
	tflearn.is_training(False, session=sess)
	print 'While Calculating Val Epoch Loss, training mode is: ' + str(sess.run(tflearn.get_training_mode()))
	batches = len(models)/VAL_BATCH_SIZE
	val_loss, val_chamfer, val_forward, val_backward = 0.,0.,0.,0.
	for b in xrange(batches):
		batch_ip, batch_gt = fetch_batch(models, b, VAL_BATCH_SIZE)
		L,C,F,B,_summ = sess.run([loss,chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled, summ], feed_dict={pcl_in:batch_ip, pcl_gt:batch_gt})
		val_loss += L
		val_chamfer += C
		val_forward += F
		val_backward += B	
	return val_loss, val_chamfer, val_forward, val_backward, _summ


if __name__=='__main__':

	# Create a folder for experiment and copy the training file
	create_folder(exp_dir)
	train_filename = basename(__file__)
	os.system('cp %s %s'%(train_filename, exp_dir))
	with open(join(exp_dir, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	train_models, val_models = get_shapenet_models(FLAGS)
	batches = len(train_models) / BATCH_SIZE

	### Create placeholders
	pcl_in = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IN_PCL_SIZE, 3), name='pcl_in')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUT_PCL_SIZE, 3), name='pcl_gt')

	# Build graph
	with tf.variable_scope('densePCR'):
		out = densenet(pcl_in)

	# Scale output and gt for val losses
	out_scaled, pcl_gt_scaled = scale(pcl_gt, out)

	# Calculate Chamfer Metrics
	dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pcl_gt, out)
	dists_forward=tf.reduce_mean(dists_forward)
	dists_backward=tf.reduce_mean(dists_backward)
	chamfer_distance = dists_backward + dists_forward

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled,_, dists_backward_scaled,_ = tf_nndistance.nn_distance(pcl_gt_scaled, out_scaled)
	dists_forward_scaled=tf.reduce_mean(dists_forward_scaled)
	dists_backward_scaled=tf.reduce_mean(dists_backward_scaled)
	chamfer_distance_scaled = dists_backward_scaled + dists_forward_scaled

	# Loss
	loss = chamfer_distance

	# Training and Val data
	print '_'*30, ' DONE  loading models ', '_'*30
	batches = len(train_models) / BATCH_SIZE

	train_vars = [var for var in tf.global_variables() if 'densePCR' in var.name]

	# Optimizer
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss, var_list=train_vars)

	# Training params
	start_epoch = 0
	max_epoch = 1500

	# Define Logs Directories
	snapshot_folder = join(exp_dir, 'snapshots')
	best_folder = join(exp_dir, 'best')
	logs_folder = join(exp_dir, 'logs')
	log_file = join(exp_dir, 'logs.txt')

	# Define savers to load and store models
	saver = tf.train.Saver(max_to_keep=50)

	# Define summary variables
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('dists_forward_scaled_%d'%OUT_PCL_SIZE, dists_forward_scaled)
	tf.summary.scalar('dists_backward_scaled_%d'%OUT_PCL_SIZE, dists_backward_scaled)
	tf.summary.scalar('chamfer_distance_scaled_%d'%OUT_PCL_SIZE, chamfer_distance_scaled)
	summ = tf.summary.merge_all()

	# Create log directories
	create_folder(snapshot_folder)
	create_folder(logs_folder)
	create_folder(join(snapshot_folder, 'best'))
	create_folder(best_folder)
	
	# GPU configurations
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(config=config) as sess:

		print 'Session started'
		train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
		val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

		print 'running initializer'
		sess.run(tf.global_variables_initializer())
		print 'done'

		# Load previous checkpoint
		ckpt = tf.train.get_checkpoint_state(snapshot_folder)
		if ckpt is not None:
			print ('loading '+ckpt.model_checkpoint_path + '  ....')
			saver.restore(sess, ckpt.model_checkpoint_path)
			start_epoch = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))

		ind = 0
		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!', '*'*30

		PRINT_N = FLAGS.print_n

		if start_epoch == 0:
			with open(log_file, 'w') as f:
				f.write(' '.join(['Epoch','Train_loss','Train_chamfer','Train_fwd','Train_bkwd','Val_loss','Val_chamfer','Val_fwd','Val_bkwd','Minutes','Seconds','\n']))

		for i in xrange(start_epoch, max_epoch):
			###
			tflearn.is_training(True, session=sess)
			print 'While Training, training mode is: ' + str(sess.run(tflearn.get_training_mode()))

			np.random.shuffle(train_models)
			train_epoch_loss = 0.
			train_epoch_chamfer = 0.
			train_epoch_forward = 0.
			train_epoch_backward = 0.

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt = fetch_batch(train_models, b, BATCH_SIZE)
				L, C, F, B, _ = sess.run([loss, chamfer_distance, dists_forward, dists_backward, optim], feed_dict={pcl_in:batch_ip, pcl_gt:batch_gt})

				train_epoch_loss += L/batches
				train_epoch_chamfer += C/batches
				train_epoch_forward += F/batches
				train_epoch_backward += B/batches

				if global_step % PRINT_N == 0:
					_summ = sess.run(summ, feed_dict={pcl_in:batch_ip, pcl_gt:batch_gt})
					train_writer.add_summary(_summ, global_step)
					time_elapsed = time.time() - since
					print 'Iter = {}  Minibatch = {}  Time = {:.0f}m {:.0f}s  Loss = {:.6f}  C: {:.6f}  F: {:.6f}  B: {:.6f}'.format(
						global_step, b, time_elapsed//60, time_elapsed%60, L, C, F, B)

			if FLAGS.multiple_local:
				print 'Saving Model ....................'
				saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
				print '..................... Model Saved'
			else:
				if i % 5 == 0 and i!=0:
					print 'Saving Model ....................'
					saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
					print '..................... Model Saved'

			# Val metrics
			val_epoch_loss, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, _summ = get_epoch_loss(val_models)
			val_writer.add_summary(_summ, global_step)

			time_elapsed = time.time() - since

			with open(log_file, 'a') as f:
				epoch_str = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.0f} {:.0f}'.format(i, train_epoch_loss, train_epoch_chamfer, train_epoch_forward, train_epoch_backward, val_epoch_loss, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60)
				f.write(epoch_str+'\n')

			# Update best model if necessary
			best_val_loss = update_best(saver, sess, snapshot_folder, best_folder, val_epoch_loss, best_val_loss, best_folder)

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'TRAIN Loss: {:6f}  Chamfer: {:.6f}  Forward: {:.6f}  Backward: {:.6f}\nVAL Loss: {:.6f}  Chamfer: {:.6f}  Forward: {:.6f}  Backward: {:.6f}\nTime:{:.0f}m {:.0f}s'.format(
				train_epoch_loss, train_epoch_chamfer, train_epoch_forward, train_epoch_backward, 
				val_epoch_loss, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, 
				time_elapsed//60, time_elapsed%60)
			print 'Best val loss so far: {:.6f}'.format(best_val_loss)
			print '-'*140
			print
