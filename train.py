from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

from model.GEFPN import GEFPN
from model.grad import *
from data_NUDT import *
class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        self.grad = Get_gradient_nopadding()
        self.gradmask  = Get_gradientmask_nopadding()
        # Read image index from TXT
        # Preprocess and load data

        trainset = MyEdgeSet(mode='train')
        testset = MyEdgeSet(mode='val')
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'GRSL3':
            model       = GEFPN()

        model           = model.cuda()
        #model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model


        self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]
        self.bce = FocalLoss()

    def training(self, epoch):
        losses = []
        tbar = tqdm(self.train_data)
        self.model.train()

        for i, (data, data_sobel, labels) in enumerate(tbar):
            data = data.cuda()
            data_sobel = self.grad(data_sobel.cuda())

            labels = labels.cuda()

            if args.deep_supervision == 'True':
                pred = self.model(data, data_sobel)
                loss = 1 * SoftIoULoss(pred, labels)

            else:
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            tbar.set_description(
                'Epoch %d, training loss %.4f' % (epoch, np.mean(losses), ))
        self.train_loss = np.mean(losses)

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = []
        losses_aux = []
        losses_main = []
        with torch.no_grad():
            for i, (data, data_sobel, labels) in enumerate(tbar):
                data = data.cuda()
                data_sobel = self.grad(data_sobel.cuda())
                labels = labels.cuda()

                if args.deep_supervision == 'True':
                    pred = self.model(data, data_sobel)
                    loss = 1 * SoftIoULoss(pred, labels)

                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                losses.append(loss.item())
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f,' % (
                epoch, np.mean(losses), mean_IOU, ))
                test_loss = np.mean(losses)

            if mean_IOU >= self.best_iou:
                self.best_iou = mean_IOU
                save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                           self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        # save high-performance model


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)

if __name__ == "__main__":
    args = parse_args()
    main(args)
