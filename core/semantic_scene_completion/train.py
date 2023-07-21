from __future__ import division
# from asyncore import write
import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from config import config
from dataloader import get_train_loader
from network import Network
from nyu import NYUv2

import sys
sys.path.append('../../furnace/')
from utils.init_func import init_weight, group_weight_V1
from engine.lr_policy import WarmUpPolyLR, PolyLR
from engine.engine import Engine
from engine.logger import get_logger
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from eval import SegEvaluator
from dataloader import ValPre


'''
  multi-gpu
'''
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())
dist.init_process_group(backend="nccl")
print('world_size', dist.get_world_size())

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

logger = get_logger()
s3client = None

GPU_LIST = config.device
with Engine(custom_parser=parser) as engine:
    cudnn.benchmark = True
    seed = config.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_loader, train_sampler = get_train_loader(engine, NYUv2, s3client)

    tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    generate_tb_dir = config.tb_dir + '/tb'
    if (args.local_rank == 0):
        writer = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    norm_layer = torch.nn.BatchNorm3d
    model = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                    norm_layer=norm_layer).cuda(args.local_rank)
   
    init_weight(model, nn.init.kaiming_normal_,norm_layer, config.bn_eps, config.bn_momentum,mode='fan_in')
    base_lr = config.lr
   
    params_list = group_weight_V1(model, base_lr )
    optimizer = torch.optim.SGD(params_list,lr=base_lr,momentum=config.momentum,weight_decay=config.weight_decay)
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
   
    '''
    device = torch.device('cuda:'+str(config.device))
    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            # model.cuda()
            model = DistributedDataParallel(model)
            model.to(device)
    else:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         model = DataParallelModel(model, device_ids=engine.devices)
        # model = DistributedDataParallel(model, device_ids=engine.devices, broadcast_buffers=False, find_unused_parameters=True)
    model.to(device)
    '''    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)    
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    engine.register_state(dataloader=train_loader, model=model,optimizer=optimizer)
    print('begin train')
    best_iou = 0.
    best_miou = 0.
    best_iou_epoch = 0
    best_iou_epoch_miou = 0.
    best_miou_epoch = 0
    best_miou_epoch_iou = 0.

    for epoch in range(engine.state.epoch, config.nepochs):
        model.train()
        if torch.distributed.get_world_size() > 1:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        sum_loss = 0
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()
            lll = minibatch['lll']
            tsdf= minibatch['tsdf']
            www = minibatch['www']
            ttt = minibatch['ttt']
            mmm = minibatch['mmm']

            lll = lll.cuda(non_blocking=True)
            ttt = ttt.cuda(non_blocking=True)
            www = www.cuda(non_blocking=True)
            mmm = mmm.cuda(non_blocking=True)
            tsdf = tsdf.cuda(non_blocking=True)

            output, boutput = model(ttt, tsdf, mmm)
            cri_weights = 10 * torch.FloatTensor([0.010820392313388523, 0.4585814244886793, 0.0411831291920445, 0.04826918042332931, 0.33162496143513115, 0.22373353821746247, 3*0.09748478737233816, 0.1478032329336482, 0.16258443136359715, 1.0, 0.0254366993244824, 0.05126348601814224])
            
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none', weight=cri_weights).cuda()
            selectindex = torch.nonzero(www.view(-1)).view(-1)
            filterLabel = torch.index_select(lll.view(-1), 0, selectindex) # [-1]
            filterOutput = torch.index_select(output.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex) # [-1, 12]
            loss_semantic = criterion(filterOutput, filterLabel)
            loss_semantic = torch.mean(loss_semantic)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            loss = loss_semantic
            loss.backward()
            sum_loss += loss.item()  

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (sum_loss / (idx + 1))

            pbar.set_description(print_str, refresh=False)
        if (args.local_rank == 0):
            writer.add_scalar("SemanticLoss/Epoch", loss_semantic.item(), epoch)
            writer.add_scalar("TrainingLoss/Epoch", sum_loss / len(pbar), epoch)


        sc = None
        ssc = None
        val_loss = 0.
       
        if ((epoch < 200 and ((epoch % config.eval_epoch == 0) or (epoch == config.nepochs - 1)))or epoch >=200)and (args.local_rank == 0):

            data_setting = {'i_root': config.i_root_folder,
                    'g_root': config.g_root_folder,
                    'h_root':config.h_root_folder,
                    'm_root': config.m_root_folder,
                    't_source': config.t_source,
                    'e_source': config.e_source}
            val_pre = ValPre()
            dataset = NYUv2(data_setting, 'val', val_pre)
            with torch.no_grad():
                all_dev = [args.local_rank]
                
                segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, model,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, False)
                sc, ssc, val_loss = segmentor.run(model, epoch, config.val_log_file, config.link_val_log_file)
                writer.add_scalar("Val Loss / Epoch", val_loss, epoch)
                writer.add_scalar("meanIOU", ssc[2], epoch)
                writer.add_scalar("ssc_prec.", ssc[3], epoch)
                writer.add_scalar("IOU", sc[0], epoch)
                writer.add_scalar("sc_prec.", sc[1], epoch)
                writer.add_scalar("sc_recall", sc[2], epoch)
               
                if ssc[2] >= best_miou:
                    best_miou = ssc[2]
                    best_miou_epoch = epoch
                    best_miou_epoch_iou = sc[0]
                    engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link,
                                                'best_miou.pth')
                if sc[0] >= best_iou:
                    best_iou = sc[0]
                    best_iou_epoch = epoch
                    best_iou_epoch_miou = ssc[2]
                    engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link,
                                                'best_iou.pth')
            best_iou_miou_file = os.path.join(tb_dir, 'best_iou_miou')
            with open(best_iou_miou_file, 'w') as f:
                f.write("best_iou_epoch: " + str(best_iou_epoch) + "\n")
                f.write("best_iou: " + str(best_iou) + "\n")
                f.write("best_iou_epoch_miou: " + str(best_iou_epoch_miou) + "\n")

                f.write("best_miou_epoch: " + str(best_miou_epoch) + "\n")
                f.write("best_miou: " + str(best_miou) + "\n")
                f.write("best_miou_epoch_iou: " + str(best_miou_epoch_iou) + "\n")
    
 
    if (dist.get_world_size() > 1) and (args.local_rank == 0): 
        writer.close()
    
    