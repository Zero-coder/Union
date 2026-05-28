def init_distributed_mode(args):
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0


def is_main_process():
    return True


def get_world_size():
    return 1


def gather_tensors_from_all_gpus(tensor):
    return tensor
