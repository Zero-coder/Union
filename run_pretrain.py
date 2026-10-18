"""Masked reconstruction pretraining and checkpoint evaluation."""
from exp.exp_pretrain import Exp_All_Task
from run import build_parser as supervised_parser, experiment_setting, seed_everything, validate_args


def build_parser():
    parser = supervised_parser()
    parser.description = "Union masked reconstruction pretraining"
    parser.set_defaults(task_data_config_path="data_provider/multi_task_pretrain.yaml",
                        lradj="cosine", min_lr=1e-6)
    parser.add_argument("--right_prob", type=float, default=1.0)
    parser.add_argument("--min_mask_ratio", type=float, default=0.5)
    parser.add_argument("--max_mask_ratio", type=float, default=0.8)
    parser.add_argument("--min_keep_ratio", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if args.right_prob != 1.0 or args.min_keep_ratio is not None:
        raise ValueError("Only independent element masking is supported; right_prob/min_keep_ratio variants are not implemented.")
    if not 0 < args.min_mask_ratio <= args.max_mask_ratio < 1:
        raise ValueError("Mask ratios must satisfy 0 < min_mask_ratio <= max_mask_ratio < 1.")
    if not 0 < args.beta2 < 1 or args.eps <= 0:
        raise ValueError("Invalid AdamW beta2/eps.")
    base_seed = args.fix_seed
    for repetition in range(args.itr):
        args.fix_seed = base_seed + repetition
        seed_everything(args.fix_seed)
        setting = experiment_setting(args, repetition)
        print("Pretraining %s; seed=%d" % (setting, args.fix_seed))
        exp = Exp_All_Task(args)
        if args.is_training:
            exp.train(setting)
            exp.test(setting)
        else:
            exp.test(setting, load_pretrain=True)


if __name__ == "__main__":
    main()
