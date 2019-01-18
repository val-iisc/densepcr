'''
Code for training the baseline approach, which consists of directly predicting 16k points.
No emd calculated anywhere because computationally expensive; also no reg loss; this is difference from train_baseline.py
Run as:
	python train_end_to_end.py --ip 251 --gpu 0 --exp 1_fg --exp_psgn expts/baseline/1_1024_emd --exp_dense_1 expts/upsampling/1a_fg_1024/ --exp_dense_2 expts/upsampling/1b_fg_4096/ --category all --bottleneck 512 --fold
	python train_end_to_end.py --ip 251 --gpu 0 --exp 2_fgl --exp_psgn expts/baseline/1_1024_emd --exp_dense_1 expts/upsampling/2a_fgl_1024/ --exp_dense_2 expts/upsampling/2b_fgl_4096/ --category all --bottleneck 512 --fold --local
'''

from importer import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_imgs', type=str, required=True, 
	help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--exp_base', type=str, required=True, 
	help='Name of base Experiment')
parser.add_argument('--exp_dense_1', type=str, required=True, 
	help='Name of dense 1024-4096 Experiment')
parser.add_argument('--exp_dense_2', type=str, required=True, 
	help='Name of dense 4096-16384 Experiment')
parser.add_argument('--bottleneck', type=int, required=True, default=128, 
	help='latent space size')
parser.add_argument('--category', type=str, required=True, 
	help='Category to visualize from : \
	["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", "monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--fold', action='store_true', 
	help='Supply this parameter if you want to use local grid folding, otherwise ignore')
parser.add_argument('--local', action='store_true', 
	help='Supply this parameter if you want to use local point feature, otherwise ignore')
parser.add_argument('--multiple_local', action='store_true', 
	help='Supply this parameter if you want to use multiple local point features, otherwise ignore')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--bn', action='store_true', 
	help='Supply this parameter if you want batchnorm in the network, otherwise ignore')
parser.add_argument('--load_best', action='store_true', 
	help='load best val model according to chamfer')
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

exp_dir = join('expts','end_to_end',FLAGS.exp)


def fetch_batch(models, indices, batch_num, batch_size):

	batch_ip = []
	batch_gt = {1024:[],4096:[],16384:[]}

	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:

		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])

		pcl_gt = {}
		pcl_gt[16384] = np.load(join(FLAGS.data_dir_pcl, model_path, pcl_16k_fname))
		pcl_gt[4096] = np.load(join(FLAGS.data_dir_pcl, model_path, pcl_4k_fname))
		pcl_gt[1024] = pcl_gt[4096][:1024]
		ip_image = cv2.imread(img_path)[4:-5,4:-5,:3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)		
		batch_ip.append(ip_image)
		for stage in hierarchies:
			batch_gt[stage].append(pcl_gt[stage])

	batch_ip = np.array(batch_ip)
	for stage in hierarchies:
		batch_gt[stage] = np.array(batch_gt[stage])

	return batch_ip, batch_gt


def get_epoch_loss(models, indices):
	tflearn.is_training(False, session=sess)
	print 'While Calculating Val Epoch Loss, training mode is: ' + str(sess.run(tflearn.get_training_mode()))

	batches = len(indices)/VAL_BATCH_SIZE
	val_loss = 0.
	val_chamfer, val_forward, val_backward, val_emd = {},{},{},{}
	for stage in hierarchies:
		val_chamfer[stage] = 0.
		val_forward[stage] = 0.
		val_backward[stage] = 0.
		if stage==1024:
			val_emd[stage] = 0.
	for b in xrange(batches):
		batch_ip, batch_gt = fetch_batch(models, indices, b, VAL_BATCH_SIZE)
		feed_dict = {img_inp:batch_ip}
		for stage in hierarchies:
			feed_dict[pcl_gt[stage]] = batch_gt[stage]
		L,C,F,B,E,_summ = sess.run([loss,chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled, emd_scaled, summ], feed_dict=feed_dict)
		val_loss += L
		for stage in hierarchies:
			val_chamfer[stage] += C[stage]/batches
			val_forward[stage] += F[stage]/batches
			val_backward[stage] += B[stage]/batches
			if stage==1024:
				val_emd[stage] += E[stage]/batches
	
	return val_loss, val_chamfer, val_forward, val_backward, val_emd, _summ


if __name__=='__main__':

	# Create a folder for experiment and copy the training file
	create_folder(exp_dir)
	train_filename = basename(__file__)
	os.system('cp %s %s'%(train_filename, exp_dir))
	with open(join(exp_dir, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS, NUM_VIEWS)
	batches = len(train_pair_indices) / BATCH_SIZE

	### Create placeholders
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt = {}
	for stage in hierarchies:
		pcl_gt[stage] = tf.placeholder(tf.float32, shape=(BATCH_SIZE, stage, 3), name='pcl_gt_%d'%stage)

	# Generate Prediction
	out = {}
	with tf.variable_scope('psgn_vars'):
		out[1024] = basenet(img_inp)
	with tf.variable_scope('densePCR'):
		out[4096] = densenet(out[1024])
		out[16384] = densenet(out[4096])

	base_vars = [var for var in tf.global_variables() if 'psgn_vars' in var.name]
	dense_1_vars = [var for var in tf.global_variables() if 'densenet_1024to4096' in var.name]
	dense_2_vars = [var for var in tf.global_variables() if 'densenet_4096to16384' in var.name]
	train_vars = base_vars + dense_1_vars + dense_2_vars

	out_scaled, pcl_gt_scaled = {},{}
	dists_forward, dists_backward, chamfer_distance = {},{},{}
	dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = {},{},{}
	match, emd = {},{}
	match_scaled, emd_scaled = {},{}

	for stage in hierarchies:
		# Scale output and gt for val losses
		out_scaled[stage], pcl_gt_scaled[stage] = scale(pcl_gt[stage], out[stage])

		# Calculate Chamfer Metrics
		dists_forward[stage],_,dists_backward[stage],_=tf_nndistance.nn_distance(pcl_gt[stage], out[stage])
		dists_forward[stage]=tf.reduce_mean(dists_forward[stage])
		dists_backward[stage]=tf.reduce_mean(dists_backward[stage])
		chamfer_distance[stage] = dists_backward[stage] + dists_forward[stage]

		# Calculate Chamfer Metrics on scaled prediction and GT
		dists_forward_scaled[stage],_, dists_backward_scaled[stage],_ = tf_nndistance.nn_distance(pcl_gt_scaled[stage], out_scaled[stage])
		dists_forward_scaled[stage]=tf.reduce_mean(dists_forward_scaled[stage])
		dists_backward_scaled[stage]=tf.reduce_mean(dists_backward_scaled[stage])
		chamfer_distance_scaled[stage] = dists_backward_scaled[stage] + dists_forward_scaled[stage]

		if stage==1024:
			# Calculate EMD
			match[stage] = approx_match(out[stage], pcl_gt[stage])
			emd[stage] = tf.reduce_mean(match_cost(out[stage], pcl_gt[stage], match[stage]))

			# Calculate EMD scaled
			match_scaled[stage] = approx_match(out_scaled[stage], pcl_gt_scaled[stage])
			emd_scaled[stage] = tf.reduce_mean(match_cost(out_scaled[stage], pcl_gt_scaled[stage], match_scaled[stage]))

	# Loss
	loss = 0.5*(chamfer_distance[1024]+scale_emd(emd[1024],1024))+ chamfer_distance[4096] + chamfer_distance[16384]

	# Training and Val data
	print '_'*30, ' DONE  loading models ', '_'*30
	batches = len(train_pair_indices) / BATCH_SIZE

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
	saver = tf.train.Saver(max_to_keep=50, var_list=train_vars)
	saver_base = tf.train.Saver(var_list=base_vars)
	saver_dense_1 = tf.train.Saver(var_list=dense_1_vars)
	saver_dense_2 = tf.train.Saver(var_list=dense_2_vars)

	# Define summary variables
	tf.summary.scalar('loss', loss)
	for stage in hierarchies:
		tf.summary.scalar('dists_forward_scaled_%d'%stage, dists_forward_scaled[stage])
		tf.summary.scalar('dists_backward_scaled_%d'%stage, dists_backward_scaled[stage])
		tf.summary.scalar('chamfer_distance_scaled_%d'%stage, chamfer_distance_scaled[stage])
		if stage==1024:
			tf.summary.scalar('emd_%d'%stage, emd[stage])
			tf.summary.scalar('emd_scaled_%d'%stage, emd_scaled[stage])	
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

		# Load pretrained models
		if start_epoch == 0:
			load_previous_checkpoint(join(FLAGS.exp_base, 'snapshots'), saver_base, sess)
			load_previous_checkpoint(join(FLAGS.exp_dense_1, 'snapshots'), saver_dense_1, sess)
			load_previous_checkpoint(join(FLAGS.exp_dense_2, 'snapshots'), saver_dense_2, sess)

		ind = 0
		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!', '*'*30

		PRINT_N = FLAGS.print_n

		if start_epoch == 0:
			with open(log_file, 'w') as f:
				f.write(' '.join(['Epoch','Train_loss','Train_chamfer','Train_fwd','Train_bkwd','Train_emd','Val_loss','Val_chamfer','Val_fwd','Val_bkwd','Val_emd','Minutes','Seconds','\n']))

		for i in xrange(start_epoch, max_epoch):
			###
			tflearn.is_training(True, session=sess)
			print 'While Training, training mode is: ' + str(sess.run(tflearn.get_training_mode()))

			random.shuffle(train_pair_indices)
			train_epoch_loss = 0.
			train_epoch_forward = 0.
			train_epoch_backward = 0.

			train_epoch_chamfer, train_epoch_forward, train_epoch_backward, train_epoch_chamfer, train_epoch_emd = {},{},{},{},{}
			for stage in hierarchies:
				train_epoch_chamfer[stage] = 0.
				train_epoch_forward[stage] = 0.
				train_epoch_backward[stage] = 0.
				if stage==1024:
					train_epoch_emd[stage] = 0.

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt = fetch_batch(train_models, train_pair_indices, b, BATCH_SIZE)
				feed_dict = {img_inp:batch_ip}
				for stage in hierarchies:
					feed_dict[pcl_gt[stage]] = batch_gt[stage]
				L, C, F, B, E, _ = sess.run([loss, chamfer_distance, dists_forward, dists_backward, emd, optim], feed_dict=feed_dict)

				train_epoch_loss += L/batches
				for stage in hierarchies:
					train_epoch_chamfer[stage] += C[stage]/batches
					train_epoch_forward[stage] += F[stage]/batches
					train_epoch_backward[stage] += B[stage]/batches
					if stage==1024:
						train_epoch_emd[stage] += E[stage]/batches

				if global_step % PRINT_N == 0:
					_summ = sess.run(summ, feed_dict=feed_dict)
					train_writer.add_summary(_summ, global_step)
					time_elapsed = time.time() - since
					print 'Iter = {}  Minibatch = {}  Time = {:.0f}m {:.0f}s  Loss = {:.6f}\nC: {}\nF: {}\nB: {}\nE:{}'.format(
						global_step, b, time_elapsed//60, time_elapsed%60, L, C, F, B, E)

			print 'Saving Model ....................'
			saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
			print '..................... Model Saved'

			# Val metrics
			val_epoch_loss, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_epoch_emd, _summ = get_epoch_loss(val_models, val_pair_indices)
			val_writer.add_summary(_summ, global_step)

			time_elapsed = time.time() - since

			with open(log_file, 'a') as f:
				epoch_str = '{} {:.6f} {} {} {} {} {:.6f} {} {} {} {} {:.0f} {:.0f}'.format(i, 
					train_epoch_loss, train_epoch_chamfer, train_epoch_forward, train_epoch_backward, train_epoch_emd, 
					val_epoch_loss, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_epoch_emd, 
					time_elapsed//60, time_elapsed%60)
				f.write(epoch_str+'\n')

			# Update best model if necessary
			best_val_loss = update_best(saver, sess, snapshot_folder, best_folder, val_epoch_loss, best_val_loss, best_folder)

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'TRAIN Loss: {:6f}\nChamfer: {}\nForward: {}\nBackward: {}\nEmd: {}\nVAL Loss: {:.6f}\nChamfer: {}\nForward: {}\nBackward: {}\nEmd: {}\nTime:{:.0f}m {:.0f}s'.format(
				train_epoch_loss, train_epoch_chamfer, train_epoch_forward, train_epoch_backward, train_epoch_emd, 
				val_epoch_loss, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, train_epoch_emd, 
				time_elapsed//60, time_elapsed%60)
			print 'Best val loss so far: {:.6f}'.format(best_val_loss)
			print '-'*140
			print
