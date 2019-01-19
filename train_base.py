'''
Code for training the base reconstruction network, which consists of regressing N points from an image.
N here is set to 1024.
Run as:
	python train_base.py --exp 1_1024_emd --data_dir_imgs <path to images> --data_dir_pcl <path to point clouds> --gpu 0 --bottleneck 512 --category all --pcl_size 1024 --loss emd
'''

from importer import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--data_dir_imgs', type=str, required=True, 
	help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--bottleneck', type=int, required=True, default=128, 
	help='latent space size')
parser.add_argument('--category', type=str, required=True, 
	help='Category to train on : \
	["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", "monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--pcl_size', type=int, default=1024, 
	help='GT size for training.')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--bn', action='store_true', 
	help='Supply this parameter if you want batchnorm in the network, otherwise ignore')
parser.add_argument('--loss', type=str, required=True, 
	help='Loss to optimize on. Choose from [chamfer/emd/both]')
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
PCL_SIZE = FLAGS.pcl_size

exp_dir = join('expts','base',FLAGS.exp)


def fetch_batch(models, indices, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		pcl_path = join(FLAGS.data_dir_pcl, model_path, pcl_1k_fname)
		pcl_gt = np.load(pcl_path)
		ip_image = cv2.imread(img_path)[4:-5,4:-5,:3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
		batch_ip.append(ip_image)
		batch_gt.append(pcl_gt)
	batch_ip = np.array(batch_ip)
	batch_gt = np.array(batch_gt)
	return batch_ip, batch_gt


def get_epoch_loss(models, indices):
	tflearn.is_training(False, session=sess)
	print 'While Calculating Val Epoch Loss, training mode is: ' + str(sess.run(tflearn.get_training_mode()))

	batches = len(indices)/VAL_BATCH_SIZE
	val_chamfer = 0.
	val_forward = 0.
	val_backward = 0.
	val_emd = 0.

	for b in xrange(batches):
		batch_ip, batch_gt = fetch_batch(models, indices, b, VAL_BATCH_SIZE)
		L,F,B,E, _summ = sess.run([chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled, emd_scaled, summ], feed_dict={img_inp:batch_ip, pcl_gt:batch_gt})
		val_chamfer += L/batches
		val_forward += F/batches
		val_backward += B/batches
		val_emd += E/batches
	
	return val_chamfer, val_forward, val_backward, val_emd, _summ


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
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PCL_SIZE, 3), name='pcl_gt')

	# Build graph
	with tf.variable_scope('psgn_vars'):
		out_base = basenet(img_inp)

	# Scale output and gt for val losses
	out_scaled, pcl_gt_scaled = scale(pcl_gt, out_base)

	# Calculate Chamfer Metrics
	dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pcl_gt, out_base)
	dists_forward=tf.reduce_mean(dists_forward)
	dists_backward=tf.reduce_mean(dists_backward)
	chamfer_distance = dists_backward + dists_forward

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled,_, dists_backward_scaled,_ = tf_nndistance.nn_distance(pcl_gt_scaled, out_scaled)
	dists_forward_scaled=tf.reduce_mean(dists_forward_scaled)
	dists_backward_scaled=tf.reduce_mean(dists_backward_scaled)
	chamfer_distance_scaled = dists_backward_scaled + dists_forward_scaled

	# Calculate EMD
	match = approx_match(out_base, pcl_gt)
	emd = tf.reduce_mean(match_cost(out_base, pcl_gt, match))

	# Calculate EMD scaled
	match_scaled = approx_match(out_scaled, pcl_gt_scaled)
	emd_scaled = tf.reduce_mean(match_cost(out_scaled, pcl_gt_scaled, match_scaled))

	# Loss
	if FLAGS.loss == 'chamfer':
		loss = chamfer_distance
	elif FLAGS.loss == 'emd':
		loss = emd
	elif FLAGS.loss == 'both':
		loss = chamfer_distance + scale_emd(emd)
	else:
		sys.exit('Loss should be chamfer or emd')
	if not FLAGS.bn:
		loss_regularization = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		loss = loss + loss_regularization*0.01

	# Training and Val data
	print '_'*30, ' DONE  loading models ', '_'*30
	batches = len(train_pair_indices) / BATCH_SIZE

	train_vars = [var for var in tf.global_variables() if 'psgn' in var.name]
	base_vars = [var for var in tf.global_variables() if 'psgn' in var.name]

	# Optimizer
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss, var_list=train_vars)

	# Training params
	start_epoch = 0
	max_epoch = 1500

	# Define Logs Directories
	snapshot_folder = join(exp_dir, 'snapshots')
	best_chamfer_folder = join(exp_dir, 'best_chamfer')
	best_emd_folder = join(exp_dir, 'best_emd')
	logs_folder = join(exp_dir, 'logs')
	log_file = join(exp_dir, 'logs.txt')

	# Define savers to load and store models
	saver = tf.train.Saver(max_to_keep=2)
	saver_base = tf.train.Saver(base_vars)

	# Define summary variables
	summary_loss = tf.summary.scalar('loss', loss)
	summary_forward_scaled = tf.summary.scalar('dists_forward_scaled', dists_forward_scaled)
	summary_backward_scaled = tf.summary.scalar('dists_backward_scaled', dists_backward_scaled)
	summary_chamfer_scaled = tf.summary.scalar('chamfer_distance_scaled', chamfer_distance_scaled)
	summary_emd_scaled = tf.summary.scalar('emd_scaled', emd_scaled)
	summ = tf.summary.merge_all()

	# Create log directories
	create_folder(snapshot_folder)
	create_folder(logs_folder)
	create_folder(join(snapshot_folder, 'best_chamfer'))
	create_folder(join(snapshot_folder, 'best_emd'))
	create_folder(best_chamfer_folder)
	create_folder(best_emd_folder)
	
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
		best_val_chamfer = 10000000
		best_val_emd = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!', '*'*30

		PRINT_N = FLAGS.print_n

		if start_epoch == 0:
			with open(log_file, 'w') as f:
				f.write(' '.join(['Epoch','Train_loss','Train_fwd','Train_bkwd','Val_loss','Val_fwd','Val_bkwd','Minutes','Seconds','\n']))

		for i in xrange(start_epoch, max_epoch):
			###
			tflearn.is_training(True, session=sess)
			print 'While Training, training mode is: ' + str(sess.run(tflearn.get_training_mode()))

			random.shuffle(train_pair_indices)
			train_epoch_loss = 0.
			train_epoch_forward = 0.
			train_epoch_backward = 0.

			train_loss_PRINT_N = 0.
			train_fwd_PRINT_N = 0.
			train_bkwd_PRINT_N = 0.

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt = fetch_batch(train_models, train_pair_indices, b, BATCH_SIZE)

				L, F, B, _ = sess.run([loss, dists_forward, dists_backward, optim], feed_dict={img_inp:batch_ip, pcl_gt:batch_gt})

				train_epoch_loss += L/batches
				train_epoch_forward += F/batches
				train_epoch_backward += B/batches
				train_loss_PRINT_N += L/PRINT_N
				train_fwd_PRINT_N += F/PRINT_N
				train_bkwd_PRINT_N += B/PRINT_N

				if global_step % PRINT_N == 0:
					C, E = sess.run([chamfer_distance, emd], feed_dict={img_inp:batch_ip, pcl_gt:batch_gt})
					_summ = sess.run(summ, feed_dict={img_inp:batch_ip, pcl_gt:batch_gt})
					train_writer.add_summary(_summ, global_step)
					time_elapsed = time.time() - since
					print '%d batches: Loss = {:.6f}  Fwd = {:.6f}  Bkwd = {:.6f};  1 batch: Chamfer = {:.6f}  Emd = {:.6f}  Iter = {}  Minibatch = {}  Time = {:.0f}m {:.0f}s'.format(PRINT_N, train_loss_PRINT_N, train_fwd_PRINT_N, train_bkwd_PRINT_N, C, scale_emd(E), global_step, b, time_elapsed//60, time_elapsed%60)
					train_loss_PRINT_N = 0.
					train_fwd_PRINT_N = 0.
					train_bkwd_PRINT_N = 0.

			print 'Saving Model ....................'
			saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
			print '..................... Model Saved'

			# Val metrics
			val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_epoch_emd, _summ = get_epoch_loss(val_models, val_pair_indices)
			val_writer.add_summary(_summ, global_step)

			time_elapsed = time.time() - since

			with open(log_file, 'a') as f:
				epoch_str = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.0f} {:.0f}'.format(i, train_epoch_loss, train_epoch_forward, train_epoch_backward, val_epoch_emd, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60)
				f.write(epoch_str+'\n')

			# Update best model if necessary
			best_val_chamfer = update_best(saver, sess, snapshot_folder, best_folder, val_epoch_chamfer, best_val_chamfer, best_chamfer_folder)
			best_val_emd = update_best(saver, sess, snapshot_folder, best_folder, val_epoch_emd, best_val_emd, best_emd_folder)

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'TRAIN Loss: {:6f}  Forward: {:.6f}  Backward: {:.6f} | Val emd: {:.6f} Chamfer: {:.6f}  Forward: {:.6f}  Backward: {:.6f}  Time:{:.0f}m {:.0f}s'.format(train_epoch_loss, train_epoch_forward, train_epoch_backward, val_epoch_emd, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60)
			print 'Best chamfer so far: {:.6f}'.format(best_val_chamfer)
			print 'Best emd so far: {:.6f}'.format(best_val_emd)
			print '-'*140
			print
