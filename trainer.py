import os
import cv2
import time
import numpy as np

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils
import datasets
#from dataset import ImageRawDataset, PartialCompEvalDataset, PartialCompDataset
import inference as infer
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json

from PIL import Image

torch.autograd.set_detect_anomaly(True)

class Trainer(object):

    def __init__(self, args):

        # get rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('{}/events'.format(args.exp_path)):
                os.makedirs('{}/events'.format(args.exp_path))
            if not os.path.exists('{}/images'.format(args.exp_path)):
                os.makedirs('{}/images'.format(args.exp_path))
            if not os.path.exists('{}/logs'.format(args.exp_path)):
                os.makedirs('{}/logs'.format(args.exp_path))
            if not os.path.exists('{}/checkpoints'.format(args.exp_path)):
                os.makedirs('{}/checkpoints'.format(args.exp_path))

            # logger
            if args.trainer['tensorboard']:
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(
                    args.exp_path))
            else:
                self.tb_logger = None
                
            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(args.exp_path))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(args.exp_path))

        # create model
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain, dist_model=True)

        # optionally resume from a checkpoint
        assert not (args.load_iter is not None and args.load_pretrain is not None), \
            "load_iter and load_pretrain are exclusive."

        if args.load_iter is not None:
            self.model.load_state("{}/checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0

        self.curr_step = self.start_iter

        # args.data.val_image_root = 'data/COCOA/val2014' # originally
        args.data['val_image_root'] = '/aul/homes/byang010/attacking-amodal/COCOA/s_val2014/animal'
        print(args.data)
        
        # lr scheduler & datasets
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]

        if not args.validate:  # train
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim,
                args.model['lr_steps'],
                args.model['lr_mults'],
                args.model['lr'],
                args.model['warmup_lr'],
                args.model['warmup_steps'],
                last_iter=self.start_iter - 1)

            train_dataset = trainval_class(args.data, 'train')
            train_sampler = utils.DistributedGivenIterationSampler(
                train_dataset,
                args.model['total_iter'],
                args.data['batch_size'],
                last_iter=self.start_iter - 1)
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=args.data['batch_size'],
                                           shuffle=False,
                                           num_workers=0, # before it was args.data['workers'] and was getting a dataloader runtime error
                                           pin_memory=False,
                                           sampler=train_sampler)
        
        val_dataset = trainval_class(args.data, 'val')
        val_sampler = utils.DistributedSequentialSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=32, # val_loader for validation only
            shuffle=False,
            num_workers=0, # before it was args.data['workers'] and was getting a dataloader runtime error
            pin_memory=False,
            sampler=val_sampler)

        self.args = args

    def run(self):

        # offline validate function is called 
        if self.args.validate:
            self.validate('off_val')
            return

        if self.args.trainer['initial_val']:
            self.validate('on_val')

        # train
        self.train()

    def train(self):

        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('train')

        end = time.time()
        for i, inputs in enumerate(self.train_loader):
            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            loss_dict = self.model.step()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())

            btime_rec.update(time.time() - end)
            end = time.time()

            self.curr_step += 1

            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer[
                    'print_freq'] == 0:
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k),
                                                  recorder[k].avg,
                                                  self.curr_step)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step,
                                               len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec) + loss_str +
                    'lr {lr:.2g}'.format(lr=curr_lr))

            # save
            if (self.rank == 0 and
                (self.curr_step % self.args.trainer['save_freq'] == 0 or
                 self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/checkpoints".format(self.args.exp_path),
                    self.curr_step)

            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or
                self.curr_step == self.args.model['total_iter']):
                self.validate('on_val')

    def validate(self, phase):        
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')

        end = time.time()
        
        # accessing image info
        images_info = self.val_loader.dataset.data_reader.images_info
        with open("batch_images_used_for_masks.json", "w") as outfile:
            print('...how many images are we expecting to get masks for? ', len(images_info))
            # extract the filenames for each image
            for b in range(len(images_info)):
                img_info = images_info[b]
                json.dump(img_info['file_name'], outfile)
                outfile.write('\n')
            print('...image filenames of the batch corresponding to masks saved in file batch_images_used_for_masks.json')
        
        all_together = []
        for i, inputs in enumerate(self.val_loader):
            if ('val_iter' in self.args.trainer and self.args.trainer['val_iter'] != -1 and i == self.args.trainer['val_iter']):
                break

            dtime_rec.update(time.time() - end)
            
            # inputs is a list of length 4
            # inputs[0] has a length of size batch_size set for val_loader (right now 1000)
            # types in inputs: <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'>
            # print('types in inputs: ', type(inputs[0]), type(inputs[1]), type(inputs[2]), type(inputs[3]))
            
            # print('...inputs.data_reader.images_info', len(inputs.data_reader.images_info))
            # print('...one of them', inputs.data_reader.images_info[0:4])
            self.model.set_input(*inputs)
            
            # tensor_dict has the output of the for each val_loader input
            tensor_dict, loss_dict = self.model.forward_only(val=phase=='off_val')

            # original_images = inputs[0]
            # new_tensor_dict = {'originals': original_images}
            # tensor_dict.update(new_tensor_dict)
            
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())
            btime_rec.update(time.time() - end)
            end = time.time()

            # tb visualize
            
            # if self.rank == 0:
            # what if i try to run this for every batch
            disp_start = max(self.args.trainer['val_disp_start_iter'], 0)
            disp_end = min(self.args.trainer['val_disp_end_iter'], len(self.val_loader))
            print('...i, disp_start, disp_end: ', i, disp_start, disp_end)
            
            if (i >= disp_start and i < disp_end):
                all_together.append(utils.visualize_tensor(tensor_dict, self.args.data.get('data_mean', [0,0,0]), self.args.data.get('data_std', [1,1,1])))
             
            if (i == disp_end - 1 and disp_end > disp_start):
                all_together = torch.cat(all_together, dim=2)
                # so it seems all_together has a column of mask/boundary images, we want to get the column of original images (added)
                # this only once in this for-loop
                grid = vutils.make_grid(all_together,
                                        nrow=1,
                                        normalize=True,
                                        range=(0, 255),
                                        scale_each=False)
                print('...grid shape from validate: ', grid.shape) # grid shape is the same as all_together shape
                
                if self.tb_logger is not None:
                    self.tb_logger.add_image('Image_' + phase, grid, self.curr_step)
                cv2.imwrite("{}/images/{}_{}_{}.png".format(self.args.exp_path, phase, self.curr_step, i), grid.permute(1, 2, 0).numpy()*255)

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_val':
                    self.tb_logger.add_scalar('val_{}'.format(k),
                                              recorder[k].avg,
                                              self.curr_step)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                    k, loss=recorder[k])

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + loss_str)

        self.model.switch_to('train')