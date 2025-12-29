import os
import torch
import collections
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch.utils.data as Data

from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import KFold


class HSA_data_loader():
    def __init__(self, 
                 data_dir,
                 batch_size, 
                 n_memory=32, 
                 shuffle=True,  
                 validation_split=0.1,
                 test_split=0.2,   # train:val:test = 7：1：2
                 num_workers=1):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_memory = n_memory
        self.num_workers = num_workers
        
        '''Data loading and preprocessing'''
        self.herb_symptom_df, self.ppi_df, self.proseq_df, self.hpi_df, self.spi_df = self.load_data()  
        self.node_map_dict, self.node_num_dict = self.get_node_map_dict()
        self.df_node_remap()     
        self.feature_index = self.herb_symptom_process()

        '''create dataset'''
        self.dataset = self.create_dataset() 
        
        self.graph, self.ppi_edge_list = self.build_ppi_graph()
        
        self.symptom_protein_dict, self.herb_protein_dict = self.get_target_dict()
   
        self.pro_seq_dict = self.get_seq_dict()
       
        self.herbs = list(self.herb_protein_dict.keys())
        self.symptoms = list(self.symptom_protein_dict.keys())

        self.symptom_neighbor_set = self.get_neighbor_set(items=self.symptoms, item_target_dict=self.symptom_protein_dict)
        self.herb_neighbor_set = self.get_neighbor_set(items=self.herbs, item_target_dict=self.herb_protein_dict)
        
        '''save data '''
        self._save()
        

    def get_ppi_edge_list(self):
        return self.ppi_edge_list
    
    def get_dsi_graph(self):
        return self.dsi_graph   
    
    def get_pro_list(self):
        return self.pro_seq_dict
    
    def get_symptom_neighbor_set(self):
        return self.symptom_neighbor_set

    def get_herb_neighbor_set(self):
        return self.herb_neighbor_set

    def get_feature_index(self):
        return self.feature_index

    def get_node_num_dict(self):
        return self.node_num_dict

    
    def load_data(self):
        # 1-hsi
        herb_symptom_df = pd.read_csv(os.path.join(self.data_dir, '01_HSA.csv'))
       
        # 2-ppi
        ppi_df = pd.read_excel(os.path.join(self.data_dir, '02_PPI.xlsx'), engine='openpyxl')
        
        # 3-pro_seq
        proseq_df = pd.read_excel(os.path.join(self.data_dir, '03_ProSeq.xlsx'), engine='openpyxl')
        
        # 4-hpi
        hpi_df = pd.read_csv(os.path.join(self.data_dir, '04_HPA.csv')) 
        
        # 5-spi
        spi_df = pd.read_csv(os.path.join(self.data_dir, '05_SPA.csv'))
        
        return herb_symptom_df, ppi_df, proseq_df, hpi_df, spi_df
    
    
    def get_node_map_dict(self):
        protein_node = list(set(self.proseq_df['protein id']))
        symptom_node = list(set(self.spi_df['symptom']))
        herb_node = list(set(self.hpi_df['herb']))
        
        node_num_dict = {
            'protein': len(protein_node), 
            'symptom': len(symptom_node), 
            'herb': len(herb_node)
        }
        
        node_map_dict = {protein_node[idx]:idx for idx in range(len(protein_node))}   
        
        node_map_dict.update({herb_node[idx]:idx for idx in range(len(herb_node))})

        node_map_dict.update({symptom_node[idx]:idx for idx in range(len(symptom_node))})
    
        print('# proteins: {0}, # herbs: {1}, # symptoms: {2}'.format(len(protein_node), len(herb_node), len(symptom_node)))
        print('# protein-protein interactions: {0}, # herb-protein associations: {1}, # symptom-protein associations: {2}'.format(len(self.ppi_df), len(self.hpi_df), len(self.spi_df)))
        
        return node_map_dict, node_num_dict  

    

    def df_node_remap(self):
        # print('\\\\\\\\\\\\') 
        self.ppi_df['proteinA'] = self.ppi_df['proteinA'].map(self.node_map_dict)
        self.ppi_df['proteinB'] = self.ppi_df['proteinB'].map(self.node_map_dict)
        self.ppi_df = self.ppi_df[['proteinA', 'proteinB']]
              
        self.proseq_df['protein id'] = self.proseq_df['protein id'].map(self.node_map_dict)
        self.proseq_df.sort_values(by='protein id', ascending=True, inplace=True)
        # proseq_df --- ['protein id', 'seq']
       
        self.spi_df['symptom'] = self.spi_df['symptom'].map(self.node_map_dict)
        self.spi_df['protein'] = self.spi_df['protein'].map(self.node_map_dict)
        self.spi_df = self.spi_df[['symptom', 'protein']]

        self.hpi_df['herb'] = self.hpi_df['herb'].map(self.node_map_dict)
        self.hpi_df['protein'] = self.hpi_df['protein'].map(self.node_map_dict)
        self.hpi_df = self.hpi_df[['herb', 'protein']]

        self.herb_symptom_df['herb'] = self.herb_symptom_df['herb'].map(self.node_map_dict)
        self.herb_symptom_df['symptom'] = self.herb_symptom_df['symptom'].map(self.node_map_dict)
        
        
    def herb_symptom_process(self):
        self.herb_symptom_df.to_csv(os.path.join(self.data_dir, 'herb_symptom_process.csv'), index=False)
        self.herb_symptom_df = self.herb_symptom_df[['herb','symptom','indication']]  
 
        self.ppi_df.to_csv(os.path.join(self.data_dir, 'ppi_process.csv'), index=False)
        self.proseq_df.to_csv(os.path.join(self.data_dir, 'proseq_process.csv'), index=False)
        
        return {'herb': 0, 'symptom': 1}
 
    
    def build_ppi_graph(self):
        tuples = [tuple(x) for x in self.ppi_df.values]
        graph = nx.Graph()
        graph.add_edges_from(tuples)

        edge_info = nx.to_pandas_edgelist(graph)
        edge_list = torch.tensor(edge_info[['source', 'target']].values).t().contiguous()

        return graph, edge_list
    
    
    def get_seq_dict(self):
        proseq_dict = self.proseq_df.set_index('protein id')['Sequence'].to_dict()
        return proseq_dict


    def get_target_dict(self):
        # 1-symptom
        sp_dict = collections.defaultdict(list) 
        symptom_list = list(set(self.spi_df['symptom'])) 
        for symptom in symptom_list:
            symptom_df = self.spi_df[self.spi_df['symptom']==symptom] 
            target = list(set(symptom_df['protein'])) 
            sp_dict[symptom] = target

        # 2-herb
        hp_dict = collections.defaultdict(list)
        herb_list = list(set(self.hpi_df['herb']))
        for herb in herb_list:
            herb_df = self.hpi_df[self.hpi_df['herb']==herb]
            target = list(set(herb_df['protein']))
            hp_dict[herb] = target
            
        return sp_dict, hp_dict


    def create_dataset(self):
        # ['herb', 'symptom', 'indication']
        self.herb_symptom_df = self.herb_symptom_df.sample(frac=1, random_state=1)

        feature = torch.from_numpy(self.herb_symptom_df.to_numpy())
        label = torch.from_numpy(self.herb_symptom_df[['indication']].to_numpy())
        feature = feature.type(torch.LongTensor)
        label = label.type(torch.FloatTensor)
        
        dataset = Data.TensorDataset(feature, label)
        return dataset
    
    
    def get_neighbor_set(self, items, item_target_dict): 
        print('constructing targets set ...')
        if items is self.symptoms:
            max_pro = 200 
        elif items is self.herbs:
            max_pro = 300
            
        neighbor_set  = collections.defaultdict(list)
  
        for item in items:
            target_list = item_target_dict[item]  
            l = len(target_list)
            if l < max_pro:
                pad_l = max_pro - l 
                padded_targets = target_list + list(np.random.choice(target_list, size=pad_l, replace=True))
            else:
                padded_targets  = target_list[:max_pro]  # cut off
            neighbor_set[item] = padded_targets   
   
        return neighbor_set  
    

    def _save(self):
        with open(os.path.join(self.data_dir, 'node_map_dict.pickle'), 'wb') as f:
            pickle.dump(self.node_map_dict, f)
        with open(os.path.join(self.data_dir, 'symptom_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.symptom_neighbor_set, f)
        with open(os.path.join(self.data_dir, 'herb_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.herb_neighbor_set, f)

            
               
    def split_start(self):
        dataset = self.dataset
    
        idx_full = np.arange(len(dataset))
        np.random.shuffle(idx_full)

        test_end = int(len(dataset) * 0.1)
        val_end = test_end + int(len(dataset) * 0.1)
        test_idx = idx_full[:test_end]
        val_idx = idx_full[test_end:val_end]
        train_idx = idx_full[val_end:]
        
        inters = self.herb_symptom_df.loc[train_idx.tolist()]
       
        init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'collate_fn': default_collate,
            'num_workers': self.num_workers
        }
    
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
    
        train_loader = DataLoader(sampler=train_sampler, **init_kwargs)
        val_loader = DataLoader(sampler=val_sampler, **init_kwargs)
        test_loader = DataLoader(sampler=test_sampler, **init_kwargs)
    
        # torch.save(train_loader, '../data/warm_start/train_data_loader.pth')
        # torch.save(val_loader, '../data/warm_start/val_data_loader.pth')
        # torch.save(test_loader, '../data/warm_start/test_data_loader.pth')
        print(f"Data split complete. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return train_loader, val_loader, test_loader, inters
    
    
     
    '''5-cv'''
    
    def warm_start_5v(self):
        dataset = self.dataset

        idx_full = np.arange(len(dataset))
        np.random.shuffle(idx_full)

        print("Current working directory:", os.getcwd())
        print(self.data_dir)
       
        directory_save = os.path.join(self.data_dir, 'data_5v/warm_start')
        os.makedirs(directory_save, exist_ok=True)
        
        test_end = int(len(dataset) * 0.1)
        test_idx = idx_full[:test_end]
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_val_idx = idx_full[test_end:]
        train_val_sampler = SubsetRandomSampler(train_val_idx)

        init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'collate_fn': default_collate,
            'num_workers': self.num_workers
        }
        
        test_loader = DataLoader(sampler=test_sampler, **init_kwargs)
        torch.save(test_loader, os.path.join(directory_save, 'test_data_loader.pth'))
        
        sss = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_id, (train_index, valid_index) in enumerate(sss.split(train_val_idx)):
            train_idx = np.array(train_val_idx)[train_index]
            valid_idx = np.array(train_val_idx)[valid_index]

            inters=self.herb_symptom_df.loc[train_idx.tolist()]
            inters.to_csv(os.path.join(directory_save, 'inters_fold_'+str(fold_id)+'.txt'),index=False)
        
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
            valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
            torch.save(train_data_loader, os.path.join(directory_save, 'train_data_loader_fold_'+str(fold_id)+'.pth'))
            torch.save(valid_data_loader, os.path.join(directory_save, 'valid_data_loader_fold_'+str(fold_id)+'.pth'))
            
        # return train_data_loader, val_data_loader, test_loader, inters
        return directory_save
    
    
    
    
    
    def cold_start_5v(self):
        dataset = self.dataset
        
        idx_full = np.arange(len(dataset))
        np.random.shuffle(idx_full)

        print("Current working directory:", os.getcwd())
        print(self.data_dir)
    
        directory_save = os.path.join(self.data_dir, 'data_5v/cold_start')
        os.makedirs(directory_save, exist_ok=True)
        
        test_end = int(len(dataset) * 0.05)
        test_idx = idx_full[:test_end]
        test_sampler = SubsetRandomSampler(test_idx)
        
        herbs=list(set(self.herb_symptom_df.loc[test_idx].loc[:,'herb'].values.tolist()))
        symptoms=list(set(self.herb_symptom_df.loc[test_idx].loc[:,'symptom'].values.tolist()))
      
        train_val_idx  = self.herb_symptom_df[~self.herb_symptom_df['herb'].isin(herbs)][~self.herb_symptom_df['symptom'].isin(symptoms)].index.tolist()
        train_val_sampler = SubsetRandomSampler(train_val_idx)

        init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'collate_fn': default_collate,
            'num_workers': self.num_workers
        }
        
        test_loader = DataLoader(sampler=test_sampler, **init_kwargs)
        torch.save(test_loader, os.path.join(directory_save, 'test_data_loader.pth'))
        
        sss = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_id, (train_index, valid_index) in enumerate(sss.split(train_val_idx)):
            train_idx = np.array(train_val_idx)[train_index]
            valid_idx = np.array(train_val_idx)[valid_index]

            inters=self.herb_symptom_df.loc[train_idx.tolist()]
            inters.to_csv(os.path.join(directory_save, 'inters_fold_'+str(fold_id)+'.txt'),index=False)
        
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
            valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
            torch.save(train_data_loader, os.path.join(directory_save, 'train_data_loader_fold_'+str(fold_id)+'.pth'))
            torch.save(valid_data_loader, os.path.join(directory_save, 'valid_data_loader_fold_'+str(fold_id)+'.pth'))
            
            print(list(set(self.herb_symptom_df.loc[train_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(symptoms)))
            print(list(set(self.herb_symptom_df.loc[valid_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(symptoms)))
            print(list(set(self.herb_symptom_df.loc[train_idx.tolist()].loc[:,'herb'].values.tolist()) & set(herbs)))
            print(list(set(self.herb_symptom_df.loc[valid_idx.tolist()].loc[:,'herb'].values.tolist()) & set(herbs)))
        
            
        # return train_data_loader, val_data_loader, test_loader, inters
        return directory_save
    
    

    def cold_start_herb_5v(self):
        dataset = self.dataset
        
        print("Current working directory:", os.getcwd())
        print(self.data_dir)

        directory_save = os.path.join(self.data_dir, 'data_5v/cold_start_herb')
        os.makedirs(directory_save, exist_ok=True)
            
        herbs = list(set(self.herb_symptom_df['herb'].values.tolist()))  
        np.random.shuffle(herbs)  
        
        test_herbs = herbs[0:int(len(herbs) * 0.1)]  
        
        test_idx = self.herb_symptom_df[self.herb_symptom_df['herb'].isin(test_herbs)].index.tolist()  
        train_val_idx  = self.herb_symptom_df[~self.herb_symptom_df['herb'].isin(test_herbs)].index.tolist()
        test_sampler = SubsetRandomSampler(test_idx)
        train_val_sampler = SubsetRandomSampler(train_val_idx)

       
        init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'collate_fn': default_collate,
            'num_workers': self.num_workers
        }
        
        test_loader = DataLoader(sampler=test_sampler, **init_kwargs)
        torch.save(test_loader, os.path.join(directory_save, 'test_data_loader.pth'))
        
        sss = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_id, (train_index, valid_index) in enumerate(sss.split(train_val_idx)):
            train_idx = np.array(train_val_idx)[train_index]
            valid_idx = np.array(train_val_idx)[valid_index]
            
            inters=self.herb_symptom_df.loc[train_idx.tolist()]
            inters.to_csv(os.path.join(directory_save, 'inters_fold_'+str(fold_id)+'.txt'),index=False)
        
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
            valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
            torch.save(train_data_loader, os.path.join(directory_save, 'train_data_loader_fold_'+str(fold_id)+'.pth'))
            torch.save(valid_data_loader, os.path.join(directory_save, 'valid_data_loader_fold_'+str(fold_id)+'.pth'))
            print(list(set(self.herb_symptom_df.loc[train_idx.tolist()].loc[:,'herb'].values.tolist()) & set(test_herbs)))
            print(list(set(self.herb_symptom_df.loc[valid_idx.tolist()].loc[:,'herb'].values.tolist()) & set(test_herbs)))
          
        # return train_data_loader, val_data_loader, test_loader, inters
        return directory_save
    
    
    
 
    def cold_start_symp_5v(self):
        dataset = self.dataset
        
        print("Current working directory:", os.getcwd())
        print(self.data_dir)

        directory_save = os.path.join(self.data_dir, 'data_5v/cold_start_symp')
        os.makedirs(directory_save, exist_ok=True)
            
        symptoms = list(set(self.herb_symptom_df['symptom'].values.tolist()))  
        np.random.shuffle(symptoms)  

        test_symptoms = symptoms[0:int(len(symptoms) * 0.1)]  
        
        test_idx = self.herb_symptom_df[self.herb_symptom_df['symptom'].isin(test_symptoms)].index.tolist()  
        train_val_idx  = self.herb_symptom_df[~self.herb_symptom_df['symptom'].isin(test_symptoms)].index.tolist()
        test_sampler = SubsetRandomSampler(test_idx)
        train_val_sampler = SubsetRandomSampler(train_val_idx)

        init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'collate_fn': default_collate,
            'num_workers': self.num_workers
        }
        
        test_loader = DataLoader(sampler=test_sampler, **init_kwargs)
        torch.save(test_loader, os.path.join(directory_save, 'test_data_loader.pth'))
        
        sss = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_id, (train_index, valid_index) in enumerate(sss.split(train_val_idx)):
            train_idx = np.array(train_val_idx)[train_index]
            valid_idx = np.array(train_val_idx)[valid_index]

            inters=self.herb_symptom_df.loc[train_idx.tolist()]
            inters.to_csv(os.path.join(directory_save, 'inters_fold_'+str(fold_id)+'.txt'),index=False)
        
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
            valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
            torch.save(train_data_loader, os.path.join(directory_save, 'train_data_loader_fold_'+str(fold_id)+'.pth'))
            torch.save(valid_data_loader, os.path.join(directory_save, 'valid_data_loader_fold_'+str(fold_id)+'.pth'))
            print(list(set(self.herb_symptom_df.loc[train_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(test_symptoms)))
            print(list(set(self.herb_symptom_df.loc[valid_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(test_symptoms)))
    
        # return train_data_loader, val_data_loader, test_loader, inters
        return directory_save
