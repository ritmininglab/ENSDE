import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from edl_module import edl_network,loss_function
from sde_module import SDEFunc
from embedding_module import emb_network
import torchsde
import os
import pickle
import random
path=os.getcwd()


#Args setup
parser = argparse.ArgumentParser('E-NSDE Recommendation')
parser.add_argument('--method', type=str, choices=['euler', 'midpoint', 'euler_heun', 'heun', 'milstein', 'log_ode'], default=None)
parser.add_argument('--sde_type', type=str, choices=['ito', 'stratonovich'], default='ito')
parser.add_argument('--noise_type', type=str, choices=["diagonal", "additive", "scalar", "general"], default='diagonal')
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

if args.adjoint:
    from torchsde import sdeint_adjoint as sdeint
else:
    from torchsde import sdeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    # dataset read here
    info_data_path = path+"/datasets/user_item_rating_time_100k.pkl"
    data = pd.read_pickle(info_data_path)

    #item embedding
    with open('item_embedding.pickle', 'rb') as handle:
        item_embedding = pickle.load(handle)

    items_len = set()

    #randomly split train and test users
    test_users=list(set([random.randrange(0,len(data)) for i in (range(int(0.3 * len(data))))]))
    train_users=list(set([usr for usr in range(len(data))])-set(test_users))

    # Define embedding model
    latent_dim = 64 #128,256
    uniq_user=943
    uniq_items=1682
    emb_model = emb_network(uniq_user, uniq_items, latent_dim).to(device)

    #NSDE model
    sde_model_user = SDEFunc(args.noise_type,args.sde_type,args.method,latent_dim).to(device)
    sde_model_item = SDEFunc(args.noise_type, args.sde_type, args.method, latent_dim).to(device)

    #define edl model
    edl_model=edl_network(latent_dim * 2).to(device)

    #define parameters to update
    params = list(emb_model.parameters()) + list(sde_model_user.parameters())+list(sde_model_item.parameters())+ list(edl_model.parameters())
    # params =  list(emb_model.parameters()) +list(sde_model_user.parameters())+list(sde_model_item.parameters())

    optimizer = optim.RMSprop(params, lr=1e-2)

    # cosine similarity
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    # #Item SDE just for one update due to not much fluctuation
    itms=torch.tensor([i for i in range(uniq_items)]).to(device)
    item_embed = emb_model.item(itms)
    itm_emb_sde_rep = {}
    ts = torch.linspace(0, 1, 1)
    for i in range(len(item_embed)):
        itm_emb_out = sdeint(sde_model_item, item_embed[i].view(1, latent_dim), ts)
        itm_emb_sde_rep[i]=itm_emb_out[-1]


    #Time interval
    t_size=2
    ts=torch.linspace(0, 1, t_size)
    mse = torch.nn.MSELoss()
    topk=5
    init_num=5
    #Training here
    for itr in range(1, args.niters + 1):
        train_loss = []
        # prec_temp = []
        # ndcg_temp=[]
        final_loss=[]
        for indx, u in enumerate(train_users):
            itms=[itm[0] for itm in data[u]][init_num:]
            items = [itm[0] for itm in data[u]][init_num:]
            rating=[rat[1] for rat in data[u]][init_num:]
            target = torch.tensor(rating, dtype=torch.float32).view(1, -1).to(device)
            interaction_time = [y[2] for y in data[u]][init_num:]
            # Create negative items
            future_items = random.sample(items[init_num:], 5)
            candidate_items = list(set([i for i in range(uniq_items)]) - set([itm[0] for itm in data[u]]))

            negative_items = []
            for itm in future_items:
                sim_score = [cos(item_embedding[itm], item_embedding[ii]).detach().numpy() for ii in candidate_items]
                sort_index = np.argsort(np.array(sim_score))[:len(items)]
                for ind in sort_index:
                    if candidate_items[ind] not in negative_items:
                        negative_items.append(candidate_items[ind])
                    else:
                        while True:
                            rand_ind = random.randint(0, len(candidate_items) - 1)
                            if rand_ind not in sort_index:
                                negative_items.append(candidate_items[rand_ind])
                                break

            # Final negative items
            negative_item = negative_items[:len(items)]

            # negative_items = list(set([i for i in range(uniq_items)]) - set(items))[:len(items)]

            u = torch.tensor(u).to(device)
            itms = torch.tensor(itms).to(device)
            # make increasing time interval: can't be zero

            time_extra = [0.00001 * i for i in range(len(interaction_time))]
            integration_time = torch.tensor([time - interaction_time[0] for time in interaction_time]).float().to(device)
            integration_time = integration_time / integration_time[len(integration_time) - 1] + torch.tensor(time_extra)

            time_diff=[0]+[interaction_time[i+1]-interaction_time[i] +0.00001 for i in range(len(interaction_time)-1)]
            time_diff = torch.tensor(time_diff).float().to(device)
            time_diff = time_diff / max(time_diff)

            #Call embedding models
            user_emb, item_emb = emb_model.forward(u, itms)
            neg_emb=emb_model.item(torch.tensor(negative_item).to(device))

            #Call User SDE module
            user_init=emb_model.item(torch.tensor(items[:5]).to(device))
            user_init=torch.mean(user_init,dim=0)
            user_emb = torch.mean(torch.stack([user_init,user_emb]),dim=0)
            user_emb_sde = sdeint(sde_model_user, user_emb.view(1,latent_dim), integration_time[:5])
            user_emb_sde=user_emb_sde[-1]

            # final evolving user representation
            user_final = [user_emb_sde]

            #Call Item SDE module and evolve user representation
            itm_emb_sde=[]
            for i,item in enumerate(item_emb):
                itm_emb_out = sdeint(sde_model_item, item.view(1,latent_dim), ts)
                itm_emb_sde.append(itm_emb_out[-1])
                user_final.append(torch.mean(torch.stack([user_final[i],itm_emb_out[-1]]),dim=0))

            itm_emb_pos=torch.stack(itm_emb_sde,dim=0)

            user_emb_final=torch.stack(user_final[:-1],dim=0)


            itm_emb_sde = []
            for item in neg_emb:
                itm_emb_out = sdeint(sde_model_item, item.view(1, latent_dim), ts)
                itm_emb_sde.append(itm_emb_out[-1])

            itm_emb_neg = torch.stack(itm_emb_sde, dim=0)

            # concat user and item embed
            edl_data = torch.cat((user_emb_final[:,-1,:], itm_emb_pos[:,-1,:]), dim=1)
            neg_data = torch.cat((user_emb_final[:,-1,:], itm_emb_neg[:,-1,:]), dim=1)


            # This part is to do simulation of varied time gap
            from simulated_uncertainty_plot_with_time import ComputeSimulation
            time_simulation = ComputeSimulation(edl_model)
            time_interval = [1.0,10.0,100.0,1000.0,10000.0,100000.0]
            for time in time_interval:
                time_compute = torch.tensor([time]*2).float().to(device)
                total_uncertainty = time_simulation.compute_simulation(edl_data.data[:2], neg_data.data[:2],time_compute)
                print(total_uncertainty)

            # edl_data = torch.cat((user_emb_sde.view(-1, latent_dim).expand(len(itm_emb_pos), -1), itm_emb_pos[:,-1,:]), dim=1)
            # neg_data = torch.cat((user_emb_sde.view(-1, latent_dim).expand(len(itm_emb_neg), -1), itm_emb_neg[:, -1, :]), dim=1)
            gamma, beta, alpha, v, negative_rating, a, nu, b = edl_model.forward(edl_data, neg_data, time_diff)

            e_loss= loss_function(target.view(-1,1), gamma, v, alpha, beta )

            # compute epistemic uncertainty for positive items
            epis_uncer = beta / ((v * (alpha - 1)) + torch.exp(torch.tensor(-10)))
            epis_uncer_norm = epis_uncer / max(epis_uncer)

            # compute epistemic uncertainty for negative items
            epis_neg = b/ ((nu * (a - 1)) + torch.exp(torch.tensor(-10)))

            rmse_loss = torch.sqrt(mse(gamma, target.view(-1,1)) + 1e-6).to(torch.float32)

            # Define BPR loss function for negative items
            bpr_loss = torch.tensor(0.0)
            w_ij = []
            tau = 3
            eta=0.001

            # final ratings for negative items
            r_hat = negative_rating + eta * epis_neg
            for r in r_hat:
                if r > tau:
                    w_ij.append(0.9)
                else:
                    w_ij.append(0.2)
            for k in range(len(w_ij)):
                bpr_loss += -w_ij[k] * torch.log(torch.sigmoid(gamma[k] - negative_rating[k]))[0]

            #total loss
            zeta=0.001
            t_loss=e_loss+zeta*bpr_loss
            final_loss.append(t_loss)
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            if indx % 50 == 0:
                loss_back = torch.mean(torch.stack(final_loss))
                train_loss.append(loss_back)
                final_loss=[]
                print('Done for User {} with Total Loss:{}'.format(indx,loss_back))

        print('\n\n=================================')
        print('Epoch {} Training Loss={}'.format(itr, torch.mean(torch.stack(train_loss))))
        print('=================================\n')
