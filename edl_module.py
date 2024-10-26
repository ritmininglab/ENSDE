import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# RMSE loss
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat.view(-1,1), y) + self.eps)
        return loss

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*(mu2-mu1)**2)  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):

    error = torch.mean(torch.abs(y-gamma))

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        inv_epi = (v*(alpha-1))/beta
        reg = error*torch.mean(inv_epi)

    return torch.mean(reg) if reduce else reg


def loss_function(y,gamma,alpha,beta,v):
    lam=1e-4
    # gamma = y_pred[:, 0]
    # v = F.softplus(y_pred[:, 1])
    # alpha = F.softplus(y_pred[:, 2]) + 1
    # beta = F.softplus(y_pred[:, 3])
    nll_loss = NIG_NLL(y, gamma, v, alpha, beta)
    reg_loss = NIG_Reg(y, gamma, v, alpha, beta)
    loss = nll_loss + lam * reg_loss

    return loss #,gamma,v,alpha,beta


# EDL Network
class edl_network(nn.Module):
    def __init__(self, input_dim):
        super(edl_network, self).__init__()
        #rating network
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) #rating only
        self.fc8 = nn.Linear(64, 4) #for negative items
        #monotonic network
        # self.fc4 = PositiveLinear(input_dim + 1, 128)
        # self.fc5 = PositiveLinear(128, 64)
        # self.fc6 = PositiveLinear(64, 1)  # for beta all positive weight layers
        # self.fc7 = NegativeLinear(64, 2)  # for alpha and nu we need negative weight at last layer

        self.fc4 = PositiveLinear(1, 8)
        self.fc6 = PositiveLinear(8, 1) # for beta all positive weight layers
        # self.fc7 = PositiveLinear(8, 2) # for alpha and nu we need negative weight at last layer
        self.fc7 = NegativeLinear(8, 2)

    def forward(self, positive_data,negative_data,time_diff):
        #rating for positive data
        hidden_out = self.fc1(positive_data)
        hidden_out = F.relu(hidden_out)
        hidden_out = self.fc2(hidden_out)
        hidden_out = F.relu(hidden_out)
        positive_rating = self.fc3(hidden_out)

        hidden_out = self.fc1(negative_data)
        hidden_out = F.relu(hidden_out)
        hidden_out = self.fc2(hidden_out)
        hidden_out = F.relu(hidden_out)
        output=self.fc8(hidden_out)

        #uncertainty
        # time_data=torch.cat((input,time_diff.view(-1,1)), dim=1)
        # hidden_out = self.fc4(time_data)
        hidden_out = self.fc4(time_diff.view(-1,1))
        # hidden_out = F.relu(self.fc5(hidden_out))
        beta= F.softplus(self.fc6(hidden_out))
        alpha_v = F.softplus(self.fc7(hidden_out))
        alpha=alpha_v[:,0]+1
        alpha=alpha.view(-1,1)
        v=alpha_v[:,1]+1e-6
        v=v.view(-1,1)
        return positive_rating,beta,alpha,v ,output[:, 0],F.softplus(output[:, 1])+1,F.softplus(output[:, 2])+1e-6,F.softplus(output[:, 3])

    def predict(self, input,time_diff):
        hidden_out = self.fc1(input)
        hidden_out = F.relu(hidden_out)
        hidden_out = self.fc2(hidden_out)
        hidden_out = F.relu(hidden_out)
        output = self.fc3(hidden_out)

        #uncertainty
        # time_data=torch.cat((input,time_diff.view(-1,1)), dim=1)
        # hidden_out = self.fc4(time_data)
        hidden_out = self.fc4(time_diff.view(-1,1))
        # hidden_out = F.relu(self.fc5(hidden_out))
        beta= F.softplus(self.fc6(hidden_out))
        alpha_v = F.softplus(self.fc7(hidden_out))
        alpha=alpha_v[:,0]+1
        alpha=alpha.view(-1,1)
        v=alpha_v[:,1]+1e-6
        v=v.view(-1,1)
        return output,beta,alpha,v



class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())


class NegativeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NegativeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, -self.log_weight.exp())