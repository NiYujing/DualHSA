import numpy as np 
import pandas as pd 

import os
import gc 
from collections import Counter

import torch
from transformers import BertModel, BertTokenizer

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


class ProtBertFeaturizer:
    def __init__(self, model_path='./prot_bert', max_length=500):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_path).to(self.device)
        self.max_length = max_length
                
    def get_seq_features(self, sequences):
        protein_emb_list = []

        for seq in tqdm(sequences):
            sequence_example = ' '.join(list(seq))[:self.max_length] 
            encoded_input = self.tokenizer(sequence_example, return_tensors='pt').to(self.device)
            output = self.model(**encoded_input)

            emb = output['last_hidden_state'][:,0][0].detach().cpu()
            #print(f'emb-type---{type(emb)}')
            #print(f'emb-shape---{emb.shape}')
            protein_emb_list.append(emb)
            torch.cuda.empty_cache()
        
        return protein_emb_list   # all_protein_features
        

        
        
        
if __name__ == "__main__":
    PB_Featurizer = ProtBertFeaturizer(model_path='./prot_bert')

    # df = pd.read_excel('pro_seq.xlsx')
    seqs = df['seq'].tolist()
    seqs = seqs[:10]

    pro_embeddings_list = PB_Featurizer.get_seq_features(seqs)
    pro_embeddings_tensor = torch.stack(pro_embeddings_list)  

    torch.save(pro_embeddings_tensor, 'pro_emb.pt')
    
    np.savetxt('pro_emb.txt', pro_embeddings_tensor.numpy()) 