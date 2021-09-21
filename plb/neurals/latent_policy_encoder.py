import numpy as np
import torch
import torch.nn as nn

from .autoencoder import PCNEncoder

class LatentPolicyEncoder(nn.Module):
	def __init__(self,
				 n_particles,
				 n_layers,
				 feature_dim,
				 hidden_dim,
				 latent_dim,
				 primitive_dim):
		super(LatentPolicyEncoder,self).__init__()
		self.n_particles = n_particles
		self.n_layers = n_layers
		self.feat_dim = feature_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.primitive_dim = primitive_dim
		self.pointencoder = PCNEncoder(state_dim=self.feat_dim,latent_dim=self.latent_dim)
		for p in self.pointencoder.parameters():
			p.requires_grad = False
		self.primitiveencoder = MLP(num_layers=self.n_layers,in_dim=self.primitive_dim,hidden_dim=256,out_dim=10)
		self.output_dim = self.latent_dim*2 + self.primitiveencoder.out_dim

	def forward(self,obs):
		if obs.ndim == 1:
			state_current = torch.from_numpy(obs[:self.feat_dim*self.n_particles].reshape(-1,self.feat_dim)).float().to(device)
			state_prev = torch.from_numpy(obs[self.feat_dim*self.n_particles:self.feat_dim*self.n_particles*2].reshape(-1,self.feat_dim)).float().to(device)
			primitive_state = torch.from_numpy(obs[self.feat_dim*2*self.n_particles:]).float().to(device)
		else:
			state_current = obs[:,:self.feat_dim*self.n_particles].reshape(obs.shape[0],self.feat_dim,-1)
			state_prev = obs[:,self.feat_dim*self.n_particles:2*self.feat_dim*self.n_particles].reshape(obs.shape[0],self.feat_dim,-1)
			primitive_state = obs[:,2*self.n_particles*self.feat_dim:]
		state_current_hidden = self.pointencoder(state_current).squeeze()
		state_prev_hidden = self.pointencoder(state_prev).squeeze()
		primitive_hidden = self.primitiveencoder(primitive_state).squeeze()
		if obs.ndim == 1:
			latent = torch.cat([primitive_hidden,state_current_hidden,state_prev_hidden],dim=0)
		else:
			latent = torch.cat([primitive_hidden,state_current_hidden,state_prev_hidden],dim=1)
		return latent
	
	def load_model(self,path):
		self.pointencoder.load_state_dict(torch.load(path))
