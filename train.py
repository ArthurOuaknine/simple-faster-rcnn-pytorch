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
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

from radar_utils import download

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(seq_loader, faster_rcnn, signal_type, test_num=10000):
    carrada = download('Carrada')
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # print('*** Evaluation ***')
    for n_seq, sequence_data in tqdm(enumerate(seq_loader)):
        seq_name, seq = sequence_data
        path_to_frames = os.path.join(carrada, seq_name[0])
        frame_set = CarradaDataset(opt, seq, 'box', signal_type,
                                   path_to_frames)
        frame_loader = data_.DataLoader(frame_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=opt.num_workers)

        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(frame_loader)):
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
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
    return result


def train(**kwargs):
    opt._parse(kwargs)

    carrada = download('Carrada')
    signal_type = 'range_angle'
    # path_to_frames = os.path.join(carrada, seq_name)

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

    faster_rcnn = FasterRCNNVGG16(n_fg_class=3)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # FLAG: vis. Important ?
    # trainer.vis.text(dataset.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for n_seq, sequence_data in tqdm(enumerate(train_seqs_loader)):
            seq_name, seq = sequence_data
            path_to_frames = os.path.join(carrada, seq_name[0])
            train_frame_set = CarradaDataset(opt, seq, 'box', signal_type,
                                             path_to_frames)
            train_frame_loader = data_.DataLoader(train_frame_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers)

            for ii, (img, bbox_, label_, scale) in tqdm(enumerate(train_frame_loader)):
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                trainer.train_step(img, bbox, label, scale)

                if (ii + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    # plot loss
                    trainer.vis.plot_many(trainer.get_meter_data())

                    # plot groud truth bboxes
                    ori_img_ = carrada_inverse_normalize(at.tonumpy(img[0]), signal_type)
                    gt_img = visdom_bbox(ori_img_,
                                         at.tonumpy(bbox_[0]),
                                         at.tonumpy(label_[0]))
                    trainer.vis.img('gt_img', gt_img)

                    # plot predicti bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], signal_type,
                                                                            visualize=True)
                    pred_img = visdom_bbox(ori_img_,
                                           at.tonumpy(_bboxes[0]),
                                           at.tonumpy(_labels[0]).reshape(-1),
                                           at.tonumpy(_scores[0]))
                    trainer.vis.img('pred_img', pred_img)

                    # rpn confusion matrix(meter)
                    trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                    # roi confusion matrix
                    trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        # FLAG: eval/test
        eval_result = eval(val_seqs_loader, faster_rcnn, signal_type, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        # FLAG: eval/test
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)

        if epoch == 9 and best_map != 0:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        # if epoch == 13:
            # break


if __name__ == '__main__':
    import fire

    fire.Fire()
