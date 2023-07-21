#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
import sys

sys.path.append('../../furnace/')
from utils.pyt_utils import parse_devices

from engine.evaluatorV2 import Evaluator
from engine.logger import get_logger
from engine.seg_opr.metric import compute_score
from nyu import NYUv2
from network import Network
from dataloader import ValPre


logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        lll = data['lll']
        www = data['www']
        mmm = data['mmm']
        tsdf = data['tsdf']
        ttt = data['ttt']
        name = data['fn']
        pp, ps, pe, sc_origin = self.eval_ssc(ttt, tsdf, mmm, device)
        results_dict = {'pp':pp, 'll':lll, 'ww':www, 'pred': ps, 'tsdf':tsdf,
                        'name':name, 'mm':mmm, 'sc_origin': sc_origin}
        return results_dict

    def hist_info(self, n_cl, ppp, ggg):
        assert (ppp.shape == ppp.shape)
        k = (ggg >= 0) & (ggg < n_cl)
        labeled = np.sum(k)
        correct = np.sum((ppp[k] == ggg[k]))

        return np.bincount(n_cl * ggg[k].astype(int) + ppp[k].astype(int),
                           minlength=n_cl ** 2).reshape(n_cl,
                                                        n_cl), correct, labeled

    def compute_metric(self, results):
        hist_ssc = np.zeros((config.num_classes, config.num_classes))
        correct_ssc = 0
        labeled_ssc = 0
        
        val_loss = []
        tp_sc, fp_sc, fn_sc, union_sc, intersection_sc = 0, 0, 0, 0, 0
        for d in results:
            ppp = d['pp'].astype(np.int64)
            lll = d['ll'].astype(np.int64)
            www = d['ww'].astype(np.float32)
            mmm = d['mm'].astype(np.int64).reshape(-1)
            name= d['name']
            lll_val = torch.from_numpy(d['ll']).cuda()
            www_val = torch.from_numpy(d['ww']).cuda()
           
            sc_origin = d['sc_origin']
           
            cri_weights = 10 * torch.FloatTensor([0.010820392313388523, 0.4585814244886793, 0.0411831291920445, 0.04826918042332931, 0.33162496143513115, 0.22373353821746247, 3*0.09748478737233816, 0.1478032329336482, 0.16258443136359715, 1.0, 0.0254366993244824, 0.05126348601814224])
           
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                            weight=cri_weights).cuda()
            selectindex = torch.nonzero(www_val.view(-1)).view(-1)
            filterLabel = torch.index_select(lll_val.view(-1), 0, selectindex)
           
            filterOutput = torch.index_select(sc_origin.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
            loss_semantic = criterion(filterOutput, filterLabel)
            loss_semantic = torch.mean(loss_semantic)
            val_loss.append(loss_semantic.item())

            flat_ppp = np.ravel(ppp)          
            flat_lll = np.ravel(lll)
            tsdf = np.ravel(d['tsdf'])

            nff = np.where((www > 0))# & (tsdf>0))
            nff_ppp = flat_ppp[nff]
            nff_lll = flat_lll[nff]

            h_ssc, c_ssc, l_ssc = self.hist_info(config.num_classes, nff_ppp, nff_lll)
            hist_ssc += h_ssc
            correct_ssc += c_ssc
            labeled_ssc += l_ssc

            ooo = (mmm == 307200) & (www > 0) & (flat_lll != 255) #& (tsdf>0)
            ooo_ppp = flat_ppp[ooo]
            ooo_lll = flat_lll[ooo]

            tp_occ = ((ooo_lll > 0) & (ooo_ppp > 0)).astype(np.int8).sum()
            fp_occ = ((ooo_lll == 0) & (ooo_ppp > 0)).astype(np.int8).sum()
            fn_occ = ((ooo_lll > 0) & (ooo_ppp == 0)).astype(np.int8).sum()

            union = ((ooo_lll > 0) | (ooo_ppp > 0)).astype(np.int8).sum()
            intersection = ((ooo_lll > 0) & (ooo_ppp > 0)).astype(np.int8).sum()

            tp_sc += tp_occ
            fp_sc += fp_occ
            fn_sc += fn_occ
            union_sc += union
            intersection_sc += intersection

        score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)
        IOU_sc = intersection_sc / union_sc
        precision_sc = tp_sc / (tp_sc + fp_sc)
        recall_sc = tp_sc / (tp_sc + fn_sc)
        score_sc = [IOU_sc, precision_sc, recall_sc]
        val_loss = np.nanmean(val_loss)
        result_line = self.print_ssc_iou(score_sc, score_ssc)
        return result_line, score_sc, score_ssc, val_loss


    def eval_ssc(self, ttt, tsdf, mmm, device=None):
        sc, bsc, esc, sc_origin = self.val_func_process_ssc(ttt, tsdf, mmm, device)
        sc = sc.permute(1, 2, 3, 0) 
        softmax = nn.Softmax(dim=3)
        sc = softmax(sc)
        ddddd = sc.cpu().numpy()
        pp = ddddd.argmax(3)

        return pp, ddddd, None, sc_origin


    def val_func_process_ssc(self, ttt, tsdf, mmm, device=None):
        tsdf = np.ascontiguousarray(tsdf[None, :], dtype=np.float32)
        tsdf = torch.FloatTensor(tsdf).cuda(device)

        ttt = np.ascontiguousarray(ttt[None, :], dtype=np.float32)
        ttt = torch.FloatTensor(ttt).cuda(device)

        mmm = torch.from_numpy(np.ascontiguousarray(mmm)[None, :]).long().cuda(device)

        with torch.cuda.device(ttt.get_device()):
            self.val_func.eval()
            self.val_func.to(ttt.get_device())
            with torch.no_grad():
                sc_origin, ssc = self.val_func(ttt, tsdf, mmm)
                sc = sc_origin[0]
                sc = torch.exp(sc)
        return sc, ssc, ssc, sc_origin


    def print_ssc_iou(self, sc, ssc):
        lines = []
        lines.append('--*-- Semantic Scene Completion --*--')
        lines.append('IOU: \n{}\n'.format(str(ssc[0].tolist())))
        lines.append('meanIOU: %f\n' % ssc[2])
        lines.append('prec.: %f\n' % ssc[3])
        lines.append('')
        lines.append('--*-- Scene Completion --*--\n')
        lines.append('IOU: %f\n' % sc[0])
        lines.append('prec.: %f\n' % sc[1])
        lines.append('recall: %f\n' % sc[2])

        line = "\n".join(lines)
        print(line)
        return line

    def get_one_hot(self, label, N): ## N classes shapenet N=4, NYU N=8
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).cuda()
        ones = ones.index_select(0, label).cuda()
        size.append(N)
        return ones.view(*size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='3', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                norm_layer=nn.BatchNorm3d, eval=True)
    data_setting = {'i_root': config.i_root_folder,
                    'g_root': config.g_root_folder,
                    'h_root':config.h_root_folder,
                    'm_root': config.m_root_folder,
                    't_source': config.t_source,
                    'e_source': config.e_source}
    val_pre = ValPre()
    dataset = NYUv2(data_setting, 'val', val_pre)
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
