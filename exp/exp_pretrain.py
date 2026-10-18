"""Masked-value reconstruction pretraining on explicitly supplied datasets."""
from exp.exp_sup import Exp_All_Task as SupervisedExp


class Exp_All_Task(SupervisedExp):
    pretrain = True
