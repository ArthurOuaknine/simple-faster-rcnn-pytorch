from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
# from data.dataset import Dataset, TestDataset, inverse_normalize
from data.carrada_dataset import Carrada
from data.dataset import SequenceCarradaDataset, CarradaDataset, TestCarradaDataset, carrada_inverse_normalize
from model import FasterRCNNVGG16
from model import FasterRCNNRESNET101
from model import FasterRCNNRESNET18
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from model.utils.bbox_tools import bbox_iou

import torch
from torch.optim.lr_scheduler import ExponentialLR
from radar_utils import download
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from radar_utils.carrada.tensorboard_visualizer import TensorboardVisualizer

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def normalize(data):
    min_value = torch.min(data)
    max_value = torch.max(data)
    return torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))


def eval(seq_loader, faster_rcnn, signal_type, test_num=10000):
    carrada = download('Carrada')
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # print('*** Evaluation ***')
    for n_seq, sequence_data in tqdm(enumerate(seq_loader)):
        seq_name, seq = sequence_data
        path_to_frames = os.path.join(carrada, seq_name[0])
        frame_set = TestCarradaDataset(opt, seq, 'box', signal_type,
                                       path_to_frames)
        frame_loader = data_.DataLoader(frame_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=opt.num_workers)

        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(frame_loader)):
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            imgs = normalize(imgs)
            pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
            gt_bboxes += list(gt_bboxes_.numpy())
            gt_labels += list(gt_labels_.numpy())
            gt_difficults += list(gt_difficults_.numpy())
            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_
            # if ii == test_num: break

    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores,
                                gt_bboxes, gt_labels, gt_difficults,
                                use_07_metric=True)
    ious = bbox_iou(gt_bboxes[0], pred_bboxes[0])
    try:
        best_iou = ious.max()
    except ValueError:
        best_iou = 0

    return result, best_iou


def train(**kwargs):
    opt._parse(kwargs)

    carrada = download('Carrada')
    train_set = Carrada().get('Train')
    val_set = Carrada().get('Validation')
    test_set = Carrada().get('Test')

    train_seqs = SequenceCarradaDataset(train_set)
    val_seqs = SequenceCarradaDataset(val_set)
    test_seqs = SequenceCarradaDataset(test_set)

    train_seqs_loader = data_.DataLoader(train_seqs, \
                                         batch_size=1, \
                                         shuffle=True, \
                                         # pin_memory=True,
                                         num_workers=opt.num_workers)

    val_seqs_loader = data_.DataLoader(val_seqs,
                                       batch_size=1,
                                       shuffle=False,
                                       # pin_memory=True,
                                       num_workers=opt.num_workers)

    test_seqs_loader = data_.DataLoader(test_seqs,
                                        batch_size=1,
                                        shuffle=False,
                                        # pin_memory=True,
                                        num_workers=opt.num_workers)

    # faster_rcnn = FasterRCNNVGG16(n_fg_class=3)
    # faster_rcnn = FasterRCNNRESNET101(n_fg_class=3)
    faster_rcnn = FasterRCNNRESNET18(n_fg_class=3)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    scheduler = ExponentialLR(trainer.faster_rcnn.optimizer, gamma=0.9)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    writer_path = os.path.join(opt.logs_path, opt.model_name)
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path)
    iteration = 0
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        print('Processing epoch: {}/{}'.format(epoch, opt.epoch))
        trainer.reset_meters()
        for n_seq, sequence_data in tqdm(enumerate(train_seqs_loader)):
            seq_name, seq = sequence_data
            path_to_frames = os.path.join(carrada, seq_name[0])
            train_frame_set = CarradaDataset(opt, seq, 'box', opt.signal_type,
                                             path_to_frames)
            train_frame_loader = data_.DataLoader(train_frame_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers)

            for ii, (img, bbox_, label_, scale) in tqdm(enumerate(train_frame_loader)):
                iteration += 1
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                img = normalize(img)

                if opt.debug_step and (iteration+1) % opt.debug_step == 0:
                    trainer.train_step(img, bbox, label, scale, stop=True)
                else:
                    trainer.train_step(img, bbox, label, scale)

                if (iteration + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    train_results = trainer.get_meter_data()
                    writer.add_scalar('Losses/rpn_loc', train_results['rpn_loc_loss'],
                                      iteration)
                    writer.add_scalar('Losses/rpn_cls', train_results['rpn_cls_loss'],
                                      iteration)
                    writer.add_scalar('Losses/roi_loc', train_results['roi_loc_loss'],
                                      iteration)
                    writer.add_scalar('Losses/roi_cls', train_results['roi_cls_loss'],
                                      iteration)
                    writer.add_scalar('Losses/total', train_results['total_loss'],
                                      iteration)

                if (iteration + 1) % opt.img_every == 0:
                    ori_img_ = at.tonumpy(img[0])
                    gt_img = visdom_bbox(ori_img_,
                                         at.tonumpy(bbox_[0]),
                                         at.tonumpy(label_[0]))
                    gt_img_grid = make_grid(torch.from_numpy(gt_img))
                    writer.add_image('Ground_truth_img', gt_img_grid, iteration)

                    # plot predicti bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], opt.signal_type,
                                                                            visualize=True)
                    # FLAG: vis
                    pred_img = visdom_bbox(ori_img_,
                                           at.tonumpy(_bboxes[0]),
                                           at.tonumpy(_labels[0]).reshape(-1),
                                           at.tonumpy(_scores[0]))
                    pred_img_grid = make_grid(torch.from_numpy(pred_img))
                    writer.add_image('Predicted_img', pred_img_grid, iteration)

                    if opt.train_eval and (iteration + 1) % opt.train_eval == 0:
                        train_eval_result, train_best_iou = eval(train_seqs_loader, faster_rcnn,
                                                                 opt.signal_type)
                        writer.add_scalar('Train/mAP', train_eval_result['map'],
                                          iteration)
                        writer.add_scalar('Train/Best_IoU', train_best_iou,
                                          iteration)

        eval_result, best_val_iou = eval(val_seqs_loader, faster_rcnn, opt.signal_type,
                                         test_num=opt.test_num)
        writer.add_scalar('Validation/mAP', eval_result['map'],
                          iteration)
        writer.add_scalar('Validation/Best_IoU', best_val_iou,
                          iteration)
        lr_ = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr_, iteration)

        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        print(log_info)
        if eval_result['map'] > best_map:
            test_result, test_best_iou = eval(test_seqs_loader, faster_rcnn, opt.signal_type,
                                              test_num=opt.test_num)
            writer.add_scalar('Test/mAP', test_result['map'],
                              iteration)
            writer.add_scalar('Test/Best_IoU', test_best_iou,
                              iteration)
            best_map = eval_result['map']
            best_test_map = test_result['map']
            best_path = trainer.save(best_val_map=best_map, best_test_map=best_test_map)
            # best_path = trainer.save(best_map=best_map)

        if (epoch + 1) % opt.lr_step == 0:
            scheduler.step()


if __name__ == '__main__':
    import fire

    fire.Fire()
