import torch as t
import warnings

class BaseConfig(object):
    # task
    gpu_ids = [0, 1]
    isTrain = True
    continue_train = False

    # training
    lr = 0.0001
    lr_policy = 'linear'
    n_epochs = 100
    n_epochs_decay = 100
    epoch_count = 1
    continue_train = False
    batch_size = 8

    # vis
    verbose = False

    # model
    load_iter = 0
    epoch = 'latest'

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)