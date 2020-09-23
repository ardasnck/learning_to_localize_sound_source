from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
		self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
		self.parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor for learning rate')
		self.parser.add_argument('--tensorboard', type=str, default='tensorboard', help='use tensorboard to visualize loss change ')
		self.parser.add_argument('--measure_time', type=bool, default=False, help='measure time of different steps during training')
		self.parser.add_argument('--niter', type=int, default=1, help='# of epochs to train, set to 1 because we are doing random sampling from the whole dataset')
		self.parser.add_argument('--num_batch', default=30000, type=int, help='number of batches to train')			
		self.parser.add_argument('--validation_on', type=bool, default=True, help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=500, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=10, help='number of batches to test for validation')
		self.parser.add_argument('--validation_visualization', type=bool, default=False, help='whether save validation predictions')
		self.parser.add_argument('--num_visualization_examples', type=int, default=20, help='number of examples to visualize')		
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=True, help='whether to augment input audio/image')

		#model arguments
		#ADD something to select different encoders/decoders
		#ADD something for additional losses

		#optimizer arguments
		self.parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate for training')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[10000, 20000], help='steps to drop LR in training samples')
		self.parser.add_argument('--optimizer', default='sgd', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0001, type=float, help='weights regularizer')
		self.mode = 'train'
