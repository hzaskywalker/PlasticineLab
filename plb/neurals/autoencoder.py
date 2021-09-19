import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self,in_dim,hidden_dim,out_dim,num_layers,activation=F.relu):
		super(MLP,self).__init__()
		assert(num_layers>=2)
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim
		self.num_layers = num_layers
		self.activation = activation
		self.layers = nn.ModuleList()
		self.layers.append(nn.Linear(self.in_dim,self.hidden_dim))
		for i in range(self.num_layers-2):
			self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim))
		self.layers.append(nn.Linear(self.hidden_dim,self.out_dim))

	def forward(self,x):
		for i in range(self.num_layers-1):
			x = self.activation(self.layers[i](x))
		x = self.layers[-1](x)
		return x

class MLPEncoder(nn.Module):
    def __init__(self,
                 n_layers,
                 dim_list):
        super(MLPEncoder,self).__init__()
        assert len(dim_list) - 1 == n_layers
        self.n_layers = n_layers
        self.dim_list = dim_list
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(dim_list[i],dim_list[i+1]))

    def forward(self,state):
        hidden = state
        for i in range(self.n_layers):
            hidden = self.layers[i](hidden).relu()
        return hidden

class MLPDecoder(nn.Module):
    def __init__(self,
                 n_layers,
                 dim_list):
        super(MLPDecoder,self).__init__()
        assert len(dim_list)-1 == n_layers
        self.n_layers = n_layers
        self.dim_list = dim_list
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(dim_list[i],dim_list[i+1]))

    def forward(self,hidden):
        for i in range(self.n_layers-1):
            hidden = self.layers[i](hidden).relu()
        out = self.layers[-1](hidden)
        return out

class MLPAutoEncoder(nn.Module):
    def __init__(self,
                 n_layers,
                 dim_list,
                 encoder,
                 decoder):
        super(MLPAutoEncoder,self).__init__()
        self.n_layers = n_layers
        self.dim_list = dim_list
        self.encoder = encoder(self.n_layers,dim_list)
        self.decoder = decoder(self.n_layers,list(reversed(self.dim_list)))

    def forward(self,x,v):
        state = torch.cat([x.flatten(),v.flatten()])
        hidden = self.encoder(state)
        state_hat = self.decoder(state)
        x_hat = state_hat[:x.flatten().size(0)].view(x.shape)
        v_hat = state_hat[x.flatten().size(0):].view(v.shape)
        return state_hat

class PointNetEncoder(nn.Module):
    def __init__(self,
                 n_particles,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(PointNetEncoder,self).__init__()
        assert(n_layers > 2)
        self.n_particles = n_particles
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(self.in_dim,self.hidden_dim))
        for _ in range(self.n_layers-2):
            self.mlps.append(nn.Linear(self.hidden_dim,self.hidden_dim))
        self.mlps.append(nn.Linear(self.hidden_dim,self.out_dim))

    # Feats: [N, feat_dim]
    # In case of batch training... feats: [batch_size,N,feat_dim]
    def forward(self,feats):
        if feats.dim() == 3:
            hidden = feats.view(-1,feats.size(2))
        else:
            hidden = feats
        for i in range(self.n_layers):
            hidden = self.mlps[i](hidden).relu()
        if feats.dim() == 3:
            hidden = hidden.view(feats.size(0),feats.size(1),-1)
            hidden = torch.max(hidden,1)[0]
        else:
            hidden = torch.max(hidden,0)[0] # hidden: [batch_size,hidden_dim]
        return hidden
            
class PointNetAutoEncoder(nn.Module):
    def __init__(self,
                 n_particles,
                 hidden_dim,
                 latent_dim,
                 feat_dim = 6,
                 n_layers = 5):
        super(PointNetAutoEncoder,self).__init__()
        self.n_particles = n_particles
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.encoder = PointNetEncoder(
            n_particles = self.n_particles,
            n_layers = self.n_layers,
            in_dim = self.feat_dim ,
            hidden_dim = self.hidden_dim,
            out_dim = self.latent_dim)

        
        self.decoder = MLPDecoder(
            3,
            [self.latent_dim,1000,10000,self.feat_dim*self.n_particles]
        )
        


    # Assume x and v are [N,3]
    def forward(self,x,v):
        x = x.squeeze()
        v = v.squeeze()
        feat = torch.cat([x,v],dim=1)
        latent = self.encoder(feat)
        out = self.decoder(latent)
        x_hat = out[:3*self.n_particles].view(-1,3)
        v_hat = out[3*self.n_particles:].view(-1,3)
        return x_hat, v_hat

class PCNEncoder(nn.Module):
    def __init__(self,state_dim=3,latent_dim=1024):
        super(PCNEncoder,self).__init__()
        self.state_dim = state_dim
        self.conv1 = nn.Conv1d(state_dim, 128, 1)
        self.conv2 = nn.Conv1d(128,256,1)
        self.conv3 = nn.Conv1d(512,512,1)
        self.conv4 = nn.Conv1d(512,latent_dim,1)

    def forward(self,x):
        input_dim = x.dim()
        if input_dim == 2:
            x = x.view(1,x.size(0),x.size(1))
            x = x.permute(0,2,1).contiguous()
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x,2)
        x = torch.cat([x,global_feature.view(batch_size,-1,1).repeat(1,1,num_points).contiguous()],1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature,_ = torch.max(x,2)
        if input_dim == 2:
            ret = global_feature.squeeze()
        else:
            ret = global_feature.view(batch_size,-1)
        return ret

class PCNDecoder(nn.Module):
    def __init__(self, feat_dim,latent_dim,n_particles):
        super(PCNDecoder,self).__init__()
        self.num_coarse = n_particles
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(self.latent_dim,1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, n_particles*self.feat_dim)

        self.conv1 = nn.Conv1d(self.feat_dim+self.latent_dim, 512,1)
        self.conv2 = nn.Conv1d(512,512,1)
        self.conv3 = nn.Conv1d(512,self.feat_dim,1)

    def forward(self,x):
        input_dim = x.dim()
        if input_dim == 1:
            x = x.view(1,-1)
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1,self.feat_dim, self.num_coarse)
        point_feat = coarse
        global_feat = x.unsqueeze(2).repeat(1,1,self.num_coarse)
        feat = torch.cat([point_feat,global_feat],1)
        center = coarse
        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        fine = fine.permute(0,2,1).squeeze()
        return fine

class PCNAutoEncoder(nn.Module):
    def __init__(self,
                 n_particles,
                 hidden_dim,
                 latent_dim,
                 feat_dim=3,
                 n_layers = None):
        super(PCNAutoEncoder,self).__init__()
        self.n_particles = n_particles
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = PCNEncoder(self.feat_dim,self.latent_dim)
        self.decoder = PCNDecoder(self.feat_dim,self.latent_dim,self.n_particles)

    def forward(self,x,v=None):
        input_dim = x.dim()
        if v != None:
            if input_dim == 3:
                feat = torch.cat([x,v],dim=2)
            else:
                feat = torch.cat([x,v],dim=1)
        else:
            feat = x

        hidden = self.encoder(feat)
        state_hat = self.decoder(hidden)
        if v != None:
            if input_dim == 3:
                x_hat = state_hat[:,:,:3]
                v_hat = state_hat[:,:,3:]
            else:
                x_hat = state_hat[:,:3]
                v_hat = state_hat[:,3:]
            return x_hat,v_hat
        else:
            return state_hat


if __name__ == "__main__":
    feat = torch.rand(10000,3)
    encoder = PCNEncoder(3,1024)
    out1 = encoder(feat)
    out2 = encoder(feat)
    print(out1.shape)
    print(out2.shape)