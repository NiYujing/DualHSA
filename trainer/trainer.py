import numpy as np
import torch
from torch_geometric.utils import from_networkx
from utils import inf_loop, MetricTracker # 文件 ./utils/util.py

from trainer.base_trainer import BaseTrainer  # 文件 ./base/base_trainer


class Trainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 criterion, 
                 metric_fns, 
                 optimizer, 
                 config, 
                 data_loader,
                 feature_index,  # 药物-症状的关联--绑定标签0/1
                 hsa_edge,
                 herb_neighbor_set,
                 symptom_neighbor_set,   
                 pro_seq_dict,
                 ppi_edge_list,
                 alpha,
                 valid_data_loader=None, 
                 test_data_loader=None, 
                 lr_scheduler=None, 
                 len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        # for data
        self.data_loader = data_loader
        self.feature_index = feature_index

        self.HSA_edge = hsa_edge
        self.herb_neighbor_set = herb_neighbor_set    
        self.symptom_neighbor_set = symptom_neighbor_set
       
        self.pro_seq_dict = pro_seq_dict
        self.PPI_edge_list = ppi_edge_list
        
        self.alpha = alpha
        
        # 确定训练时使用的迭代方式
        if len_epoch is None:  
            # epoch-based training --- 基于 epoch 的训练方式
            self.len_epoch = len(self.data_loader)# (self.data_loader[0])  # (self.data_loader)
        else:
            # iteration-based training  --- ？？？
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        
        # 设置验证集和测试集的数据加载器
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None  # 是否有验证
        self.lr_scheduler = lr_scheduler  # 学习率调度器
        # self.log_step = int(np.sqrt(data_loader[0].batch_size))
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        # 初始化 train和val的指标跟踪器
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    ''''''
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()  # 设置模型为训练模式
        self.train_metrics.reset()  # 重置train_metrics,记录当前epoch的指标
    
        '''
        print(len(self.data_loader.))
        nyj_dataloader_epoch = 
        epoch_sampler = nyj_dataloader_epoch.epoch_dataset()
        self.data_loader.sampler = epoch_sampler
       
        lenth = len(self.data_loader[epoch]) 
        
        print(f'第{epoch}个------------------长度{lenth}')
        '''
       
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(self.data_loader[epoch])
        for batch_idx, (data, target) in enumerate(self.data_loader):    # (self.data_loader[epoch]):    # (self.data_loader):  
            # data_loader中的每个batch_idx     
            '''
            (data,
             target)
            ''' 
            target = target.to(self.device)
            
            if(batch_idx == 1):
                print(data[:10])
                # print(target[:10])
            
            # print(data[:10])
            
            '''
            # 无别的loss
            output = self.model(*self._get_feed_dict(data))  
            loss = self.criterion(output, target)  
            '''
            # 有info_nce_loss
            output, info_nce_loss = self.model(*self._get_feed_dict(data))
            # output, info_nce_loss, output_gcn, output_pro = self.model(*self._get_feed_dict(data))
            # pred_loss = (self.criterion(output_pro, target.squeeze()) + self.criterion(output_gcn, target.squeeze()))/2
            class_loss = self.criterion(output, target.squeeze())
            # alpha = 0.1
            loss = self.alpha * class_loss + (1 - self.alpha) * info_nce_loss #+ 0.2 * pred_loss
            # ce_loss = self.criterion(output, target.squeeze())
            # loss = ce_loss + (0.1 * info_nce_loss) / (ce_loss.detach() + 1e-8)    
            
            # score, att, emb_loss = self.model(*self._get_feed_dict(data))
            # loss = self.criterion(output, target) + 0.01*emb_graph_loss
            # loss = self.criterion(output, target.squeeze()) + emb_loss
           
            
            self.optimizer.zero_grad()
            loss.backward()  # 反向传播
            self.optimizer.step()

            # 更新训练指标跟踪器
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(y_pred, y_true))
            
            if(batch_idx == 1):
                print(y_pred[:10])
            
            # 记录日志信息
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            
            # 一个epoch结束
            if batch_idx == self.len_epoch:
                break
        
        print('---------------------train over -----------------')
        # 计算并记录该 epoch 的训练指标数据
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        # 验证
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_'+k : v for k, v in val_log.items()}

        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()   # 设置模型为验证模式
        self.valid_metrics.reset()  # 重置valid_metrics,记录当前epoch的指标
        with torch.no_grad():  # 不计算梯度
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                # 
                target = target.to(self.device)
                '''
                print('val')
                if(batch_idx == 1):
                    print(data[:10])
                print('val')
                '''
                
                '''
                # 无别的loss
                output = self.model(*self._get_feed_dict(data))  
                loss = self.criterion(output, target)  
                '''
                # 有info_nce_loss    
                output, info_nce_loss = self.model(*self._get_feed_dict(data))
                # output, info_nce_loss, output_gcn, output_pro = self.model(*self._get_feed_dict(data))
         
                # pred_loss = (self.criterion(output_pro, target.squeeze()) + self.criterion(output_gcn, target.squeeze()))/2
                class_loss = self.criterion(output, target.squeeze())
                # alpha = 0.1
                loss = self.alpha * class_loss + (1 - self.alpha) * info_nce_loss #+ 0.2 * pred_loss 
                # ce_loss = self.criterion(output, target.squeeze())
                # loss = ce_loss + (0.1 * info_nce_loss) / (ce_loss.detach() + 1e-8)    
            
                # score, att, emb_loss = self.model(*self._get_feed_dict(data))
                # loss = self.criterion(output, target) + 0.01*emb_graph_loss
                # loss = self.criterion(output, target.squeeze()) + emb_loss
                
                
                # 更新验证指标跟踪器
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(y_pred, y_true))
                
                if(batch_idx == 1):
                    print(y_pred[:10])
                    print(y_true[:10])
                    
                
        # ？？？ add histogram of model parameters to the tensorboard---将模型参数的直方图添加到 TensorBoard 日志中
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self):
        self.model.eval()
        total_loss = 0.0
        # 创建一个与指标函数数量相同长度的张量 total_metrics，用于存储各个指标的累积值
        total_metrics = torch.zeros(len(self.metric_fns))
        nyj_p_out = []
        yuan_y = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                target = target.to(self.device)
                '''
                print('##############################')
                print(len(data))
                print(data[:20])
                print(target[:20])
                '''
                if(batch_idx == 0):
                    print('==================test0================')
                    print(data[:10])     
                
               
                '''
                # 无别的loss
                output = self.model(*self._get_feed_dict(data))  
                loss = self.criterion(output, target)  
                '''
                # 有info_nce_loss    
                output, info_nce_loss = self.model(*self._get_feed_dict(data))
                # output, info_nce_loss, output_gcn, output_pro = self.model(*self._get_feed_dict(data))
                
               
                # pred_loss = (self.criterion(output_pro, target.squeeze()) + self.criterion(output_gcn, target.squeeze()))/2
                class_loss = self.criterion(output, target.squeeze())
                # alpha = 0.1
                loss = self.alpha * class_loss + (1 - self.alpha) * info_nce_loss #+ 0.2 * pred_loss
                # ce_loss = self.criterion(output, target.squeeze())
                # loss = ce_loss + (0.1 * info_nce_loss) / (ce_loss.detach() + 1e-8)    
            
                # score, att, emb_loss = self.model(*self._get_feed_dict(data))
                # loss = self.criterion(output, target) + 0.01*emb_graph_loss
                # loss = self.criterion(output, target.squeeze()) + emb_loss
                # loss函数在config里配置，output可以是未激活也可以是激活的，这里output还未激活
                
                # ？？？
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                y_pred = torch.sigmoid(output)  # 激活
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for i, metric in enumerate(self.metric_fns):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size
                
                if(batch_idx == 0):
                    print('-----test----打分')
                    print(y_pred[:10])
                nyj_p_out.append(y_pred)
                yuan_y.append(y_true)           
        
        '''
        import csv
        flat_list = [item for sublist in nyj_p_out for item in sublist]
        # 指定要写入的文件路径
        csv_file_path = './NYJ_DSI_Result/1119_quchong_result/HXZQS/test_p.csv'
        print(csv_file_path)
        # 使用 csv 模块写入 CSV 文件
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow('p')
            writer.writerows(flat_list)
            
         
        flat_y = [item for sublist in yuan_y for item in sublist]
        # 指定要写入的文件路径
        csv_file_path2 = './NYJ_DSI_Result/1119_quchong_result/HXZQS/test_yuan_y.csv'
        print(csv_file_path2)
        # 使用 csv 模块写入 CSV 文件
        with open(csv_file_path2, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow('y')
            writer.writerows(flat_y)
        '''
        
        # total = None
        # total_metrics = None
        test_output = {'n_samples': len(self.test_data_loader.sampler), 
                       'total_loss': total_loss, 
                       'total_metrics': total_metrics}
        
        
        # test_output =None
        return test_output
    
    
    
    def test_val(self):
        self.model.eval()
        total_loss = 0.0
        # 创建一个与指标函数数量相同长度的张量 total_metrics，用于存储各个指标的累积值
        total_metrics = torch.zeros(len(self.metric_fns))
      
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                target = target.to(self.device)

                '''
                # 无别的loss
                output = self.model(*self._get_feed_dict(data))  
                loss = self.criterion(output, target)  
                '''
                # 有info_nce_loss    
                output, info_nce_loss = self.model(*self._get_feed_dict(data))
                # output, info_nce_loss, output_gcn, output_pro = self.model(*self._get_feed_dict(data))
                
                # pred_loss = (self.criterion(output_pro, target.squeeze()) + self.criterion(output_gcn, target.squeeze()))/2
                class_loss = self.criterion(output, target.squeeze())
                # alpha = 0.1
                loss = self.alpha * class_loss + (1 - self.alpha) * info_nce_loss #+ 0.2 * pred_loss
                # ce_loss = self.criterion(output, target.squeeze())
                # loss = ce_loss + (0.1 * info_nce_loss) / (ce_loss.detach() + 1e-8)    
            
                # score, att, emb_loss = self.model(*self._get_feed_dict(data))
                # loss = self.criterion(output, target) + 0.01*emb_graph_loss
                # loss = self.criterion(output, target.squeeze()) + emb_loss
                # loss函数在config里配置，output可以是未激活也可以是激活的，这里output还未激活
                
                # ？？？
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                y_pred = torch.sigmoid(output)  # 激活
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for i, metric in enumerate(self.metric_fns):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size

        test_val_output = {'n_samples': len(self.valid_data_loader.sampler), 
                       'total_loss': total_loss, 
                       'total_metrics': total_metrics}
        return test_val_output

    
     
    
    
    
    
    
    def get_save(self, save_files):
        result = dict()
        for key, value in save_files.items():
            if type(value) == dict:
                temp = dict()
                for k,v in value.items():
                    temp[k] = v.cpu().detach().numpy()
            else:
                temp = value.cpu().detach().numpy()
            result[key] = temp
        return result

    def _get_feed_dict(self, data):
        '''输入trainer的数据
        herbs.to(self.device),     ---batch内的herbs
        symptoms.to(self.device),  ---batch内的symptoms
        herb_feat,                 ---全部herbs
        symp_feat,                 ---全部symptoms
        hsa_egde,                  ---仅用于train的hsa_edge
        herbs_neighbors,           ---herbs靶标
        symptoms_neighbors,        ---symptoms相关蛋白
        pro_seq_dict,              ---protein的序列信息
        ppi_edge_list.to(self.device)    ---ppi的全部edge
        '''
        
        # [batch_size]
        herbs = data[:, self.feature_index['herb']]
        symptoms = data[:, self.feature_index['symptom']]
        
        # all
        herb_feat=torch.LongTensor(list(self.herb_neighbor_set.keys())).to(self.device)
        symp_feat=torch.LongTensor(list(self.symptom_neighbor_set.keys())).to(self.device)

        
        # dsi_egde
        dsi_egde = torch.LongTensor(self.HSA_edge).to(self.device)
        
        # 邻居蛋白
        symptoms_neighbors, herbs_neighbors = [], []
        
        herbs_neighbors = torch.LongTensor([self.herb_neighbor_set[d] for d in herbs.numpy()]).to(self.device)
        symptoms_neighbors = torch.LongTensor([self.symptom_neighbor_set[s] for s in symptoms.numpy()]).to(self.device)
            
        # pro encode
        pro_seq_dict = self.pro_seq_dict
        ppi_edge_list = self.PPI_edge_list
     
        return herbs.to(self.device), symptoms.to(self.device), herb_feat, symp_feat, dsi_egde, herbs_neighbors, symptoms_neighbors, pro_seq_dict, ppi_edge_list.to(self.device)
    
    
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
