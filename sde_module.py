import torch.nn as nn
class SDEFunc(nn.Module):

    def __init__(self,noise_type,sde_type,method,latent_dim):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.method = method

        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            nn.Tanh(),
            nn.Linear(2*latent_dim, latent_dim),
        )

        for m in self.drift_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.diff_net = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.Tanh(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

        for m in self.diff_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    #drift
    def f(self, t, y):
        return self.drift_net(y)

    #diffusion
    def g(self, t, y):
        return self.diff_net(y)