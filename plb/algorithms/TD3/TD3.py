import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...neurals.autoencoder import PCNEncoder, MLP, PointNetEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

# This Layer will merge the different observations from the env.obs to one latent state
class SRLLayer(nn.Module):
	def __init__(self,
				 n_particles,
				 n_layers,
				 feature_dim,
				 hidden_dim,
				 latent_dim,
				 primitive_dim,
				 ):
		super(SRLLayer,self).__init__()
		self.n_particles = n_particles
		self.latent_dim = latent_dim
		self.n_layers = n_layers
		self.feat_dim = feature_dim
		self.hid_dim = hidden_dim
		self.primitive_dim = primitive_dim
		self.encoder = PointNetEncoder(n_particles = self.n_particles,
									   n_layers = self.n_layers,
									   in_dim = self.feat_dim,
									   hidden_dim = self.hid_dim,
									   out_dim = self.latent_dim)
		# Freeze the SRL
		for param in self.encoder.parameters():
			param.requires_grad = False

		self.primitive_encoder = MLP(num_layers=self.n_layers,
									 in_dim = self.primitive_dim,
									 hidden_dim = 256,
									 out_dim = 10)
		self.output_dim = self.latent_dim + self.primitive_encoder.out_dim
	
	# If the Q function need state representation then this function must handle batched training
	# Without batch training obs should be a huge flatten variable
	def forward(self,obs):
		if obs.ndim == 1:
			pointcloud_state = torch.from_numpy(obs[:self.feat_dim*self.n_particles].reshape(-1,self.feat_dim)).to(device)
			primitive_state = torch.from_numpy(obs[self.feat_dim*2*self.n_particles:]).to(device)
		else:
			pointcloud_state = torch.from_numpy(obs[:,:self.feat_dim*self.n_particles].reshape(obs.shape[0],-1,self.feat_dim)).to(device)
			primitive_state = torch.from_numpy(obs[:,self.feat_dim*2*self.n_particles:]).to(device)
		primitive_hidden = self.primitive_encoder(primitive_state).squeeze()
		pointcloud_hidden = self.encoder(pointcloud_state).squeeze()
		if obs.ndim == 1:
			latent = torch.cat([primitive_hidden,pointcloud_hidden],dim=0)
		else:
			latent = torch.cat([primitive_hidden,pointcloud_hidden],dim=1)
		return latent
		
class SRLPCNLayer(nn.Module):
	def __init__(self,
				 n_particles,
				 n_layers,
				 feature_dim,
				 hidden_dim,
				 latent_dim,
				 primitive_dim):
		super(SRLPCNLayer,self).__init__()
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


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		n_particles,
		n_layers,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		enable_latent = False,
		encoder_path:str = 'pretrain_model/weight_cfm.pth'
	):
		print("state_dim:",state_dim,
			  "action_dim:",action_dim,
			  "n_particles:",n_particles)
		self.enable_latent = enable_latent
		self.state_dim = state_dim
		self.n_particles = n_particles
		self.n_layers = n_layers
		self.primitive_dim = self.state_dim - 6*self.n_particles
		if self.enable_latent:
			self.latent_encoder = SRLPCNLayer(n_particles=self.n_particles,
										      n_layers = self.n_layers,
										      feature_dim = 3,
										      hidden_dim = 256,
										      latent_dim = 1024,
										      primitive_dim = self.primitive_dim).to(device)
			self.latent_encoder.load_model(encoder_path)
			print("Enable Latent!!!",self.latent_encoder.output_dim)
			self.state_dim = self.latent_encoder.output_dim
		self.actor = Actor(self.state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(self.state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		if self.enable_latent:
			state = self.latent_encoder(state).view(1,-1).detach()
		else:
			state = torch.FloatTensor(state.reshape(1,-1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=20):
		self.total_it += 1

		# Sample replay buffer 
		latent, action, next_latent, reward, not_done = replay_buffer.sample(batch_size)\
		# Preserve the state

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_latent) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_latent, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(latent, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(latent, self.actor(latent)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor")

	def load(self, filename):
		self.actor_target = copy.deepcopy(self.actor)

	def adapt_state(self,state):
		if self.enable_latent:
			return self.latent_encoder(state).detach().cpu().numpy()
		else:
			return state
		