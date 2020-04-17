"""
Function: inherit the basetrain class and implement the actual training process
Author: Xiang.Yang
"""
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
import random
from utils.utils import auc_value, f1, cmatrix


class PredTrainer(BaseTrain):
    def __init__(self, sess, model, train_loader, eval_loader, test_loader, config, logger):
        super(PredTrainer, self).__init__(sess, model, train_loader, eval_loader, test_loader, config, logger)


    def train_epoch(self, prev_loss, epoch):
        # num_iter_per_epoch: number of batches for training
        random.shuffle(self.train_loader.dataset)
        # show the loop in a progress bar or not
        loop = tqdm(range(self.config.num_iter_per_epoch), ascii=False)
        #loop = range(self.config.num_iter_per_epoch)
        losses = []
        accs = []
        # traverse each batch in one epoch
        for idx in loop:
            loss, acc = self.train_step(idx)
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        # after training on this epoch, evaluate the model on the validation set
        # train_auc, train_f = self.evaluate_on_train_all()
        eval_loss, eval_acc, eval_auc, eval_f = self.eval_step(epoch)

        # if loss decreases, do the test
        test_loss, test_acc, test_auc, test_f = 100.0, 0.0, 0.0, 0.0
        # the loss is on the training set passed from previous epoch
        if loss < prev_loss:
            # doing test if loss decreases
            test_loss, test_acc, test_auc, test_f = self.test_step()

        # save the epoch model
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'train_loss': loss,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'eval_auc': eval_auc,
            'eval_f1': eval_f
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

        return loss, \
               eval_loss, eval_auc, eval_f, \
               test_loss, test_auc, test_f


    def train_step(self, prev_idx):
        # train on each batch
        train_inputs = next(self.train_loader.next_batch(prev_idx=prev_idx))
        random.shuffle(train_inputs)
        batch_y, batch_x, batch_pos, batch_x_len = zip(*train_inputs)
       
        feed_dict = {self.model.x: batch_x, self.model.pos: batch_pos,
                     self.model.x_len: batch_x_len,
                     self.model.y: batch_y, self.model.is_training: True,
                     }
        _, loss, acc = self.sess.run(
            [self.model.train_step, self.model.cross_entropy, self.model.accuracy],
            feed_dict=feed_dict)
    
        return loss, acc


    def eval_step(self, epoch):
        """
        evaluate by batch and then accumulate in case memory out
        :return:
        """
        # evaluate the validation set
        eval_ys, eval_probs, eval_preds, eval_accs, eval_losses = [], [], [], [], []
        eval_size = self.eval_loader.datasize
        batches = int(eval_size/self.config.batch_size) + 1
        for idx in range(batches):
            eval_inputs = next(self.eval_loader.next_batch(idx))
            batch_y, batch_x, batch_pos, batch_x_len = zip(*eval_inputs)
            
            feed_dict = {self.model.x: batch_x, self.model.pos: batch_pos,
                         self.model.x_len: batch_x_len,
                         self.model.y: batch_y, self.model.is_training: False,
                         }
            loss, acc, preds, probs, logits = self.sess.run(
                [self.model.cross_entropy, self.model.accuracy, self.model.preds, self.model.probs, self.model.logits],
                feed_dict=feed_dict)
            probs = [prob[1] for prob in probs]
                      
            eval_ys.extend(batch_y)
            eval_preds.extend(preds)
            eval_probs.extend(probs)
            eval_accs.append(acc)
            eval_losses.append(loss)

        loss = np.mean(eval_losses)
        acc = np.mean(eval_accs)

        try:
            auc = auc_value(eval_ys, eval_probs)
        except:
            print(eval_probs)

        f = f1(eval_ys, eval_preds)

        print(str(cmatrix(eval_ys, eval_preds)))
      
        return loss, acc, auc, f


    def test_step(self):
        """
        evaluate by batch and then accumulate in case memory out
        :return:
        """
        # evaluate the validation set
        test_ys, test_probs, test_preds, test_accs, test_losses = [], [], [], [], []
        test_size = self.test_loader.datasize
        batches = int(test_size / self.config.batch_size) + 1
        for idx in range(batches):
            test_inputs = next(self.test_loader.next_batch(idx))
            batch_y, batch_x, batch_pos, batch_x_len = zip(*test_inputs)
            feed_dict = {self.model.x: batch_x, self.model.pos: batch_pos,
                         self.model.x_len: batch_x_len,
                         self.model.y: batch_y, self.model.is_training: False,
                         }
            loss, acc, preds, probs = self.sess.run(
                [self.model.cross_entropy, self.model.accuracy, self.model.preds, self.model.probs],
                feed_dict=feed_dict)
            probs = [prob[1] for prob in probs]
            #
            test_ys.extend(batch_y)
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_accs.append(acc)
            test_losses.append(loss)

        loss = np.mean(test_losses)
        acc = np.mean(test_accs)
        auc = auc_value(test_ys, test_probs)
        f = f1(test_ys, test_preds)
        return loss, acc, auc, f

