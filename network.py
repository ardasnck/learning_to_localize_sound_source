import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

class AVModel(nn.Module):
	def __init__(self):
		super(AVModel, self).__init__()
		self.vision_embedding_model = VisualNet()
		self.audio_embedding_model = AudioNet()
		self.vision_embedding_model.eval()
		#######
		self.sound_fc1 = nn.Linear(1000,1000)
		self.sound_fc2 = nn.Linear(1000,512)
		self.att_softmax = nn.Softmax(dim=-1)
		self.vision_fc1 = nn.Linear(512,512)
		self.vision_fc2 = nn.Linear(512,1000)		
	def forward(self, image, pos_audio, neg_audio):
		vis_embedding = self.vision_embedding_model(image)
		pos_audio_embedding = self.audio_embedding_model(pos_audio)
		neg_audio_embedding = self.audio_embedding_model(neg_audio)
		h = self.sound_fc1(pos_audio_embedding)
		h = F.relu(h)
		h = self.sound_fc2(h)
		#L2 normalize audio tensor
		h = F.normalize(h, p=2, dim=1)
		h = h.unsqueeze(2)
		#L2 normalize channel dimension of vision embedding (N,512,20,20)
		normalized_vis_embedding = F.normalize(vis_embedding, p=2, dim=1)
		reshaped_vis_embedding = normalized_vis_embedding.view(normalized_vis_embedding.size(0), 512, 400)
		reshaped_vis_embedding = reshaped_vis_embedding.permute(0,2,1)
		##LOCALIZATION MODULE##
		att_map = torch.matmul(reshaped_vis_embedding, h)
		att_map = torch.squeeze(att_map)
		att_map = F.relu(att_map)
		att_map = self.att_softmax(att_map)
		att_map = att_map.unsqueeze(2)
		##LOCALIZATION MODULE##
		vis_embedding = vis_embedding.view(vis_embedding.size(0), 512, 400)
		z = torch.matmul(vis_embedding, att_map) #(N,512)
		z = torch.squeeze(z)
		z = F.relu(z)
		z = self.vision_fc1(z)
		z = F.relu(z)
		z = self.vision_fc2(z)
		return z, pos_audio_embedding, neg_audio_embedding, att_map


class VisualNet(nn.Module):
	def __init__(self):
		super(VisualNet, self).__init__()
		original_vgg = torchvision.models.vgg16(pretrained=True)
		for param in original_vgg.parameters():
			param.requires_grad = False
		layers = list(original_vgg.children())[0][0:29]
		self.feature_extraction = nn.Sequential(*layers) 
	def forward(self, x):
		x = self.feature_extraction(x)
		return x

class AudioNet(nn.Module): 
	def __init__(self):
		super(AudioNet, self).__init__() 

	def forward(self,x):
		x = F.relu(x)
		return x