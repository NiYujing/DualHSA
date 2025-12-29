'''
DualHSA Model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
import math

import os
from torch.nn.utils.weight_norm import weight_norm

from torch_geometric.nn import GATConv,MLP,GCNConv,SAGEConv
from torch_geometric.data import Data

from model.base_model import BaseModel  

from model.PLM.ProtBert.ProtBertFeaturizer import ProtBertFeaturizer
from model.PLM.ESM.ESM_Featurizer import ESM_Featurizer

from model.BAN import BANLayer 



class DualHSA(BaseModel):
    def __init__(self, 
                 protein_num, 
                 symptom_num,
                 herb_num,
                 emb_dim, # 64
                 l1_decay,
                 GAT_params,
                 PLM):
        super(DualHSA, self).__init__()
        self.protein_num = protein_num
        self.symptom_num = symptom_num
        self.herb_num = herb_num
        self.emb_dim = emb_dim
        self.l1_decay = l1_decay
        self.GAT_params = GAT_params
        self.PLM_params = PLM
        
        '''1 MACRO level'''
        # self.pro_emb_init = nn.Embedding(self.protein_num, self.emb_dim*2)
        self.herb_embedding = nn.Embedding(self.herb_num, self.emb_dim*2)
        self.symptom_embedding = nn.Embedding(self.symptom_num, self.emb_dim*2)
        
        self.HSA_GCN = HSA_GCN(self.emb_dim*2, self.emb_dim*4, self.emb_dim*2)
        
        self.batch_herb = nn.BatchNorm1d(self.emb_dim*2)
        self.batch_symp = nn.BatchNorm1d(self.emb_dim*2)
        
        self.combine_embedding_gcn = torch.nn.Linear(self.emb_dim*4, self.emb_dim*2)
        self.batch_gcn = nn.BatchNorm1d(self.emb_dim*2)
        
        
        '''2 MICRO level'''

        self.PPI_layers = GATMODEL(self.GAT_params,self.PLM_params['pro_dim'],concat=False)   
        
        layer_dim = self.PLM_params['pro_dim']
        self.bcn = weight_norm(
            BANLayer(v_dim=layer_dim, q_dim=layer_dim, h_dim = layer_dim*2, h_out=2), # ban_heads),
            name='h_mat', 
            dim=None)
        self.MLP_ban = MLP_ban(input_dim=layer_dim*2, hidden_dim=layer_dim//2, output_dim=self.emb_dim*2)
        
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.emb_dim*4, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 2)
        )
        
        self.MLP_score = MLPDecoder(in_dim=self.emb_dim*4, hidden_dim=self.emb_dim, out_dim=16, binary=1)
        self.infoNCE = InfoNCELoss(temperature=1.0)
       
        # self.MLP_score_gcn = MLP_gcn(self.emb_dim*2, 16, 1)
        # self.MLP_score_pro = MLP_pro(self.emb_dim*2, 16, 1)


    def forward(self, 
                herbs: torch.LongTensor,
                symptoms: torch.LongTensor, 
                herb_feat: torch.LongTensor,
                symp_feat: torch.LongTensor, 
                edge_index: torch.LongTensor,
                herb_neighbors: list, 
                symptom_neighbors: list, 
                pro_seq_dict,
                PPI_edge_list):
        
        '''1 MACRO level''' 
        x_herb = self.herb_embedding(herb_feat)
        x_herb = self.batch_herb(F.relu(x_herb))
        x_symptom = self.symptom_embedding(symp_feat)
        x_symptom = self.batch_symp(F.relu(x_symptom))

        # herb_embeddings=x_herb[herbs]
        # symptom_embeddings=x_symptom[symptoms]
        
        edge_index = edge_index.t()
        embeds_all=torch.cat((x_herb,x_symptom)) 
        embeds_all = self.HSA_GCN(embeds_all, edge_index)
        
        herb_embeddings_gcn=embeds_all[:self.herb_num][herbs]       
        symptom_embeddings_gcn=embeds_all[self.herb_num:][symptoms]   
        
        embeds_gcn=F.relu(self.combine_embedding_gcn(torch.cat([herb_embeddings_gcn,symptom_embeddings_gcn], dim=1)))
        embeds_gcn = self.batch_gcn(embeds_gcn)   
   
        
    
        '''2 MICRO level'''
        # Protein Encoder Module
        pro_embedding_plm = self.extract_PLM_emb(self.PLM_params['model_type'], pro_seq_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pro_embedding_plm = pro_embedding_plm.to(device)
        
        prot_emb_GAT, prot_attn = self.PPI_layers(pro_embedding_plm, PPI_edge_list.to(torch.long))   # GAT       

        herb_neighbors_embeddings=self.get_neighbor_embeddings(herb_neighbors, prot_emb_GAT)
        symptom_neighbors_embeddings=self.get_neighbor_embeddings(symptom_neighbors, prot_emb_GAT)
        print(f'---herb_neighbors_embeddings size：---{herb_neighbors_embeddings.shape}') # torch.Size([128, 200, 1024])
        print(f'---symptom_neighbors_embeddings size：---{symptom_neighbors_embeddings.shape}') #torch.Size([128, 100, 1024])
 
        # BAN
        f, att_pro = self.bcn(herb_neighbors_embeddings, symptom_neighbors_embeddings)
        embeds_pro = self.MLP_ban(f) # F2

        
        '''Predict'''
        info_nce_loss = self.infoNCE(embeds_gcn, embeds_pro)
  
        combined = torch.cat([embeds_pro, embeds_gcn], dim=-1)
        attention_logits = self.attention_mlp(combined)
        attention_weights = F.softmax(attention_logits, dim=-1)  
        x1_weighted = embeds_pro * attention_weights[:, 0].unsqueeze(-1)  
        x2_weighted = embeds_gcn * attention_weights[:, 1].unsqueeze(-1)
        combined_emb = torch.cat([x1_weighted, x2_weighted], dim=1)  

        score = self.MLP_score(combined_emb)
        score = torch.squeeze(score, 1)
        
        return score, info_nce_loss      
    
    
    
    def extract_PLM_emb(self, model_type, pro_seq_dict):
        if model_type == 'ESM':
            plmemb_savepath = './model/PLM/ESM_pro_embedding_plm.pt'
        elif model_type == 'ProtBert':
            plmemb_savepath = './model/PLM/PB_pro_embedding_plm.pt'
            
        seqs = list(pro_seq_dict.values())

        if os.path.exists(plmemb_savepath):    
            pro_embedding_plm = torch.load(plmemb_savepath)
            print("Loading existing PLM embedding files ...") 
        else:
            if model_type == 'ESM':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ESM_Feature = ESMFEATURE(device)
                pro_embedding_list = ESM_Feature.get_representations(seqs)
            elif model_type == 'ProtBert':
                PB_Featurizer = ProtBertFeaturizer(model_path='./model/PLM/ProtBert/prot_bert')
                pro_embedding_list = PB_Featurizer.get_seq_features(seqs)
            pro_embedding_plm = torch.stack(pro_embedding_list)
            torch.save(pro_embedding_plm, plmemb_savepath)
            print("PLM embedding have been generated and saved.")
        return pro_embedding_plm
    
     
    ''' 
    def _emb_loss(self, symptom_embeddings, herb_embeddings):
        item_regularizer = (torch.norm(symptom_embeddings) ** 2 + torch.norm(herb_embeddings) ** 2) / 2     
        emb_loss = self.l1_decay * item_regularizer / symptom_embeddings.shape[0] 
        return emb_loss
    '''
    
    def get_neighbor_embeddings(self, neighbors, pro_emb):
        print('----------------------------------------')
        print("neighbors shape:", neighbors.shape)
        neighbors_emb = pro_emb[neighbors]  # (batch_size, max_pro, dim)
        return neighbors_emb
    
  

''''''
# yes
class HSA_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HSA_GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        return x
    
    
    
# yes
class GATMODEL(nn.Module):
    def __init__(self,params,dim,concat=False):
        super(GATMODEL, self).__init__()
        if not concat:
            assert dim%params['heads'] == 0 , "Feature dimension is not divisible by heads"
            params['out_channels'] = dim//params['heads']
        self.layers = nn.ModuleList([GATConv(in_channels=-1, out_channels = params['out_channels'], heads= params['heads'], dropout=params['dropout'], add_self_loops=params['add_self_loops'])])
        
        for i in range(params['num_layers']-1):
            self.layers.append(GATConv(in_channels=params['out_channels']*params['heads'], out_channels = params['out_channels'], heads= params['heads'], dropout=params['dropout'], add_self_loops=params['add_self_loops']))
            
        self.norm = nn.LayerNorm(params['out_channels']*params['heads'])
        self.act = nn.ReLU()
        
    def forward(self, x,edge_index):
        attention_weights = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, edge_index,return_attention_weights=True)
            attention_weights.append(attn)
            #x = self.norm(x)
            #x = self.act(x)
        return x,attention_weights
    
    
    


    
'''MLP'''
# yes
class MLP_ban(torch.nn.Module): 
    def __init__(self,input_dim, hidden_dim, output_dim):
        super(MLP_ban,self).__init__()
        #两层感知机
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        # self.relu = torch.nn.ReLU()
   
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
    
# yes
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


   
    
class MLP_gcn(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MLP_gcn,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.fc23 = nn.Linear(n_hidden, n_output)
        
    def forward(self,x):
        h_1 = torch.tanh(self.fc1(x))
        h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x
            
        
class MLP_pro(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MLP_pro,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.fc23 = nn.Linear(n_hidden, n_output)
        
    def forward(self,x):
        h_1 = torch.tanh(self.fc1(x))
        h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x


    


''' Loss '''
            
# yes
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        image_embeds = F.normalize(image_embeds, dim=-1, p=2)
        text_embeds = F.normalize(text_embeds, dim=-1, p=2)

        similarity_matrix = F.cosine_similarity(image_embeds.unsqueeze(1), text_embeds.unsqueeze(0), dim=-1) / self.temperature
        sim_matrix = torch.matmul(image_embeds, text_embeds.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        nce_loss = -torch.log(torch.exp(positives) / torch.exp(similarity_matrix).sum(dim=-1)).mean()
       
        return nce_loss  