import sys, os, time
import torch
from torch import nn
from dataset import *
from torch.utils.data import DataLoader
from metrics import get_all_EERs_my
from utils import keras_decay
import pickle as pkl
import warnings
warnings.simplefilter("ignore")


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(self.args.device)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor([0.1, 0.9])
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=args.lr,
            weight_decay=0.001,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: keras_decay(step)
        )


    def training_one_iter(self, data_minibatch):
        asv1, asv2, cm2, ans, key = data_minibatch
        if torch.cuda.is_available():
            asv1 = asv1.to(self.args.device)
            asv2 = asv2.to(self.args.device)
            cm2 = cm2.to(self.args.device)
            ans = ans.to(self.args.device)

        pred = self.model(asv1, asv2, cm2)
        nloss = self.model.calc_loss(asv1, asv2, cm2, ans)
        self.optimizer.zero_grad()
        nloss.backward()
        if self.args.clip_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip_norm)
        self.optimizer.step()
        if self.args.model_name == "baseline":
            pred = torch.softmax(pred, dim=-1)
        output = (pred, key)

        return nloss, output

    def training_one_epoch(self, epoch):
        self.model.train()
        loss = 0
        preds, keys = [], []
        training_set = SASV_Dataset(self.args, "trn")
        train_loader = DataLoader(training_set, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.num_workers, drop_last=True, pin_memory=True)
        tot_batch = len(train_loader)
        for num, data_slice in enumerate(train_loader):
            nloss, output = self.training_one_iter(data_slice)
            loss += nloss
            pred, key = output
            preds.append(pred)
            keys.extend(list(key))
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%3d] Lr: %5f, " % (epoch, self.lr_scheduler.get_last_lr()[-1]) + \
                             " Training Progress: %.1f%%, Training Loss: %.5f \r" %
                             (100 * ((num + 1) / tot_batch), loss / (num + 1)))
            sys.stderr.flush()
        if self.args.model_name == "baseline" or self.args.model_name == "baseline2":
            preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        else:
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs_my(preds=preds, keys=keys)
        print("\nTrn: sasv_eer_trn: %0.3f, sv_eer_trn: %0.3f, spf_eer_trn: %0.3f" % (
            100 * sasv_eer, 100 * sv_eer, 100 * spf_eer))

        return loss / tot_batch, sasv_eer

    def validate_one_iter(self, data_minibatch):
        asv1, asv2, cm2, ans, key = data_minibatch
        if torch.cuda.is_available():
            asv1 = asv1.to(self.args.device)
            asv2 = asv2.to(self.args.device)
            cm2 = cm2.to(self.args.device)
            ans = ans.to(self.args.device)

        pred = self.model(asv1, asv2, cm2)
        nloss = self.model.calc_loss(asv1, asv2, cm2, ans)
        if self.args.model_name == "baseline":
            pred = torch.softmax(pred, dim=-1)
        output = (pred, key)

        return nloss, output

    def validate_one_epoch(self, epoch):
        self.model.eval()
        validation_set = SASV_Dataset(self.args, "dev")
        valid_loader = DataLoader(validation_set, batch_size=self.args.batch_size, shuffle=False,
                                  num_workers=self.args.num_workers, drop_last=False, pin_memory=True)
        tot_batch = len(valid_loader)
        loss = 0
        preds, keys = [], []
        with torch.no_grad():
            for num, data_slice in enumerate(valid_loader):
                nloss, output = self.validate_one_iter(data_slice)
                loss += nloss
                pred, key = output
                preds.append(pred)
                keys.extend(list(key))
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                                 " [%3d] Lr: %5f, " % (epoch, self.lr_scheduler.get_last_lr()[-1]) + \
                                 " Validating Progress: %.1f%%, Validation Loss: %.5f \r" %
                                 (100 * ((num + 1) / tot_batch), loss / (num + 1)))
                sys.stderr.flush()
            if self.args.model_name == "baseline" or self.args.model_name == "baseline2":
                preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
            else:
                preds = torch.cat(preds, dim=0).detach().cpu().numpy()
            sasv_eer, sv_eer, spf_eer = get_all_EERs_my(preds=preds, keys=keys)
            print("\nDev: sasv_eer_dev: %0.3f, sv_eer_dev: %0.3f, spf_eer_dev: %0.3f" % (
            100 * sasv_eer, 100 * sv_eer, 100 * spf_eer))

        return loss / tot_batch, sasv_eer

    def run_train(self):
        min_eer = 1e4
        init_loss, sasv_eer_dev = self.validate_one_epoch(0)
        if sasv_eer_dev < min_eer:
            torch.save(self.model, os.path.join(self.args.output_dir, "%s_best.pt"
                                                % (self.model.name)))
            min_eer = sasv_eer_dev
        try:
            for epoch_idx in range(1, (self.args.num_epochs+1)):
                train_loss, sasv_eer_trn = self.training_one_epoch(epoch_idx)
                valid_loss, sasv_eer_dev = self.validate_one_epoch(epoch_idx)
                self.lr_scheduler.step()
                save_path = os.path.join(self.args.output_dir, "checkpoints", "%s_epoch_%03d_trainloss_%.3f_valloss_%.3f_trneer_%.3f_deveer_%.3f.pt"
                                         % (self.model.name, epoch_idx, train_loss, valid_loss, sasv_eer_trn, sasv_eer_dev))
                torch.save(self.model, save_path)
                if sasv_eer_dev < min_eer:
                    torch.save(self.model, os.path.join(self.args.output_dir, "%s_best.pt"
                                         % (self.model.name)))
                    min_eer = sasv_eer_dev
        except:
            print("\nThis model will not continue training or has nothing to train.")


