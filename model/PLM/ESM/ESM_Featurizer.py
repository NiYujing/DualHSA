import torch
import esm
from tqdm import tqdm
import pandas as pd
from esm import Alphabet, ProteinBertModel
import numpy as np


class ESM_Featurizer:
    def __init__(self,device):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device
    
    def get_representations(self, data):
        max_length = 1024  
        data = [seq[:max_length] for seq in data]
        data_ = [("protein"+str(j), seq) for j, seq in enumerate(data)]
        self.model.eval()  # disables dropout for deterministic results
        
        all_rep = []
        for i, (label, seq) in tqdm(enumerate(data_), total=len(data_)):
            single_data = [(label, seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(single_data)

            batch_tokens = batch_tokens.to(self.device)
            tokens_len = (batch_tokens != self.alphabet.padding_idx).sum(dim=1) 
            # print(tokens_len)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)

            token_representations = results["representations"][33].to('cpu')
            # print(token_representations.shape)
            # print(token_representations)
            single_pro_rep = token_representations[0, 1 : tokens_len - 1].mean(0)
            # print(single_pro_rep.shape)
            # print(single_pro_rep)
            all_rep.append(single_pro_rep)
            
            del batch_tokens, results, token_representations
            torch.cuda.empty_cache()
            
        print(len(all_rep))
        return all_rep
    
    


