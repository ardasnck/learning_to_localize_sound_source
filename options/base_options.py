import argparse
import os
import torch
import shutil
#from utils import utils

def b_mkdirs(path, remove=False):
	if os.path.isdir(path):
		if remove:
			shutil.rmtree(path)
		else:
			return
	os.makedirs(path)



class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--dataset_file', default='/your_data_root/MUSICDataset/solo/')
		self.parser.add_argument('--val_dataset_file', default='/your_data_root/MUSICDataset/solo/')
		#self.parser.add_argument('--mean_path', default='/your_data_root/MUSICDataset/solo/')
		self.parser.add_argument('--annotation_path', default='/your_data_root/MUSICDataset/solo/')
		self.parser.add_argument('--dataset_path', default='/your_data_root/MUSICDataset/solo/')
		self.parser.add_argument('--class_name', default='arda')
		self.parser.add_argument('--mode', type=str, default='train', help='name of the mode. Is it training or testing?')
		#self.parser.add_argument('--duration', type=int, default=30, help='Duration of the interval')
		#self.parser.add_argument('--audio_arch', type=str, default='resnet', help='chooses how audio architecture will be constructed')
		#self.parser.add_argument('--vision_arch', type=str, default='resnet18', help='chooses how vision architecture will be constructed')
		#self.parser.add_argument('--baseline', type=str, default='baseline_first', help='chooses which model will be used')
		#self.parser.add_argument('--triplet_mining_type', type=str, default='online', help='chooses which sampling strategy dataloader will use')
		#self.parser.add_argument('--phase', type=int, default=1, help='chooses which phase of the curriculum learning will be used for training')
		#self.parser.add_argument('--class_name', default='/your_data_root/MUSICDataset/solo/')
		#self.parser.add_argument('--hdf5_path', default='/your_root/hdf5/MUSICDataset/soloduet')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='AVT_baseline_first', help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
		#self.parser.add_argument('--model', type=str, default='audioVisualMUSIC', help='chooses how datasets are loaded.')
		self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		#self.parser.add_argument('--seed', default=0, type=int, help='random seed')

		#audio arguments
		#self.parser.add_argument('--audio_window', default=400, type=int, help='audio window length')
		#self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='sound sampling rate')
		#self.parser.add_argument('--stft_no', default=512, type=int, help="number of fft")
		#self.parser.add_argument('--stft_hop', default=160, type=int, help="stft hop length")
		#self.parser.add_argument('--mels_no', default=64, type=int, help="number of mels")


		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.mode = self.mode

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		b_mkdirs(expr_dir)
		val_dir = os.path.join('val', self.opt.name)
		b_mkdirs(val_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt
