import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, train_loader, eval_loader, test_loader, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        prev_train_loss, prev_eval_auc = 100.0, 0.0
        loss_decr_counter = 0 # counter for the times of loss non decrease
        optimal_eval_epoch = -1 # the optimal evaluation epoch on auc
        optimal_eval_auc, optimal_test_auc = 0.0, 0.0
        optimal_eval_f, optimal_test_f = 0.0, 0.0

        # traverse each epoch to train, cur_epoch_tensor: the value of the current epoch number
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            cur_train_loss,  \
                eval_loss, eval_auc, eval_f, \
                test_loss, test_auc, test_f = self.train_epoch(prev_train_loss, cur_epoch)
            print('Training on epoch {} with loss {}'.format(cur_epoch, cur_train_loss)) 

            # print("\tLoss, auc, f1 on the train set are {}, {} and {}" \
            #       .format(cur_train_loss, train_auc, train_f))
            print("\tLoss, auc, f1 on the eval set are {}, {} and {}" \
                  .format(eval_loss, eval_auc, eval_f))
            print("\tLoss, auc, f1 on the test set are {}, {} and {}" \
                  .format(test_loss, test_auc, test_f))

            self.sess.run(self.model.increment_cur_epoch_tensor)
            # if loss no change, begin the counter for early termination, no change on the optimal epoch
            if cur_train_loss >= prev_train_loss:
                loss_decr_counter += 1
            # else, save the current epoch as the optimal
            else:
                loss_decr_counter = 0
            # if eval loss keeps no decrease for 5 epochs, then break and keeps the -5 epoch as optimal
            # early stop
            if loss_decr_counter >= 5:
                break
            prev_train_loss = cur_train_loss
            # select the optimal eval epoch
            if eval_auc > optimal_eval_auc:
                optimal_eval_epoch = cur_epoch
                optimal_eval_auc = eval_auc
                optimal_test_auc = test_auc
                optimal_eval_f = eval_f
                optimal_test_f = test_f
                loss_decr_counter = 0 # continue training ignoring the loss increasing
            # input()
        print("Optimal eval and test auc in Epoch {} are {}, and {}".format(optimal_eval_epoch, optimal_eval_auc,
                                                                                     optimal_test_auc))

        print("Optimal eval and test f1 in Epoch {} are {}, and {}".format(optimal_eval_epoch, optimal_eval_f,
                                                                                 optimal_test_f))


    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
