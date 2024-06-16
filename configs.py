import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Image colorization with GANs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


   
    parser.add_argument('--valid_path', type=str,
                         default='./mydata/test', help='数据集的根目录')

    parser.add_argument('--train_path', type=str, default='./mydata/train',
                        help='数据集的根目录')


    parser.add_argument('--test_path', type=str, default='./mydata/valid',
                        help='数据集的根目录')

    parser.add_argument('--batch_size', type=int, default=16, help='批处理的大小')
    parser.add_argument('--num_workers', type=int, default=4, help='批处理的大小')

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Save and load path for the network weights.')

    parser.add_argument('--epochs', type=int, default=201,
                        help='模型训练的epoch')

    # 学习率衰减的数字
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate for both networks.')
    parser.add_argument('--lr_decay_steps', type=float, default=6e4,
                        help='Learning rate decay steps for both networks.')

    return parser.parse_args()
