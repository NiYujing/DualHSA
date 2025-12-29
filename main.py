'''main'''
import argparse  
import collections 
import torch
import numpy as np

import data_loader.data_loaders as module_data   # ./data_loader/data_loaders.py
import model.loss as module_loss                 # ./model/loss.py
import model.metric as module_metric             # ./model/metric.py
from model.model import DualHSA as module_arch # ./model/model.py
from parse_config import ConfigParser            # parse_config.py
from trainer.trainer import Trainer  # ./trainer/trainer.py


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config): 
    logger = config.get_logger('train')
    HSA_data_loader = config.init_obj('HSA_data_loader', module_data)

    train_data_loader, valid_data_loader, test_data_loader, hsa_train_edge = HSA_data_loader.split_start()
    feature_index = HSA_data_loader.get_feature_index()  
    symptom_neighbor_set = HSA_data_loader.get_symptom_neighbor_set()  
    herb_neighbor_set = HSA_data_loader.get_herb_neighbor_set()  
    node_num_dict = HSA_data_loader.get_node_num_dict()  
    ppi_edge_list = HSA_data_loader.get_ppi_edge_list() 
    pro_seq_dict  = HSA_data_loader.get_pro_list()

    hsa_train_edge = hsa_train_edge.to_numpy()
    hsa_edge=hsa_train_edge[hsa_train_edge[:,-1]==1][:,:-1]   
    hsa_edge[:,1]=hsa_edge[:,1]+node_num_dict['herb']         
    
    '''Training'''
    model = module_arch(protein_num=node_num_dict['protein'],
                        symptom_num=node_num_dict['symptom'],
                        herb_num=node_num_dict['herb'],
                        emb_dim=config['arch']['args']['emb_dim'],
                        l1_decay=config['arch']['args']['l1_decay'],
                        GAT_params = config['pro_gat'],
                        PLM = config['PLM'])
    logger.info(model)
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())  
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params) 
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)  

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader= train_data_loader,  
                      feature_index=feature_index,
                      hsa_edge = hsa_edge,
                      herb_neighbor_set=herb_neighbor_set,
                      symptom_neighbor_set=symptom_neighbor_set,           
                      pro_seq_dict = pro_seq_dict,
                      ppi_edge_list = ppi_edge_list,
                      alpha = config['alpha'],
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()   
    
    '''Testing'''
    logger = config.get_logger('test')
    logger.info(model)
    test_metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume) 
    state_dict = checkpoint['state_dict']  
    model.load_state_dict(state_dict)  

    test_output = trainer.test() 
    
    log = {'loss': test_output['total_loss'] / test_output['n_samples']}
    log.update({
        met.__name__: test_output['total_metrics'][i].item() / test_output['n_samples'] \
            for i, met in enumerate(test_metrics)
    })
    logger.info(log)
    

    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='HSA_data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
