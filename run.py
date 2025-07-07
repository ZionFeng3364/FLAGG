import random
import argparse
import flgo
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.cifar100_classification as cifar100
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.femnist_classification as femnist
import flgo.benchmark.fashion_classification as fashion
import flgo.benchmark.partition as fbp
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedprox as fedprox
import flgo.algorithm.scaffold as scaffold
import flgo.algorithm.fednova as fednova
import flgo.algorithm.moon as moon
import flgo.algorithm.feddyn as feddyn
import flgo.algorithm.ditto as ditto
import algo.flatten as flatten
from models.cifar10_resnet18 import CIFAR10ResNet18
from models.cifar100_resnet18 import CIFAR100ResNet18
from utils.logger import ClassicalLogger  # 经典联邦算法需要使用该Logger进行测试和记录
from flgo.experiment.logger.pfl_logger import PFLLogger # PFL算法需要使用该Logger进行测试和记录


def parse_args():
    parser = argparse.ArgumentParser(description='联邦学习实验')

    # 数据集选择
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'femnist', 'fashion'],
                        help='数据集选择: cifar10, cifar100, mnist, femnist, fashion')

    parser.add_argument('--data_root', type=str, default='/home/c5090/data/fzj/FLAtten/dataset',
                        help='数据集根目录')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Dirichlet分布的alpha参数，控制数据分布不均衡程度')

    # 客户端相关参数
    parser.add_argument('--num_clients', type=int, default=5,
                        help='客户端数量')
    parser.add_argument('--proportion', type=float, default=1.0,
                        help='每轮参与训练的客户端比例')

    # 训练相关参数
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='联邦学习通信轮数')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='每轮本地训练的epoch数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', type=int, default=1,
                        help='使用的GPU ID')

    # 实验任务相关参数
    parser.add_argument('--algorithm', type=str, default='FLAtten',
                        choices=['FedAvg', 'FLAtten', 'FedProx', 'Scaffold', 'FedNova', 'MOON', 'FedDyn', 'FedgradMA', 'Ditto'],
                        help='联邦学习算法')
    parser.add_argument('--gen_task', action='store_true',
                        help='是否重新生成任务')
    parser.add_argument('--run', action='store_true',
                        help='是否运行训练')
    parser.add_argument('--analyze', action='store_true', default=True,
                        help='是否分析结果')
    parser.add_argument('--task_dir', type=str, default='./ex/',
                        help='任务存储目录')

    # 添加更多训练相关参数
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='本地训练的学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='本地训练的批次大小')
    parser.add_argument('--my_optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'rmsprop', 'adagrad'],
                        help='本地训练的优化器')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='本地训练的动量参数')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='本地训练的权重衰减参数')
    parser.add_argument('--clip_grad', type=float, default=0.0,
                        help='梯度裁剪阈值，大于0时生效')

    # 采样和聚合方式
    parser.add_argument('--sample', type=str, default='uniform',
                        choices=['uniform', 'md', 'full'],
                        help='客户端采样方式')
    parser.add_argument('--aggregate', type=str, default='other',
                        choices=['uniform', 'weighted_com', 'weighted_scale', 'other'],
                        help='模型聚合方式')

    # 早停和学习率衰减
    parser.add_argument('--early_stop', type=int, default=-1,
                        help='早停轮数，大于-1时生效')
    parser.add_argument('--learning_rate_decay', type=float, default=0.998,
                        help='学习率衰减系数')
    parser.add_argument('--lr_scheduler', type=int, default=-1,
                        help='全局学习率调度器类型，大于-1时生效')

    # 数据划分相关
    parser.add_argument('--train_holdout', type=float, default=0.1,
                        help='从本地训练数据中分离出验证集的比例')
    parser.add_argument('--test_holdout', type=float, default=0.0,
                        help='从服务器测试数据中分离出验证集的比例')

    # 日志选项
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='每隔多少轮评估一次')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG'],
                        help='日志级别')

    return parser.parse_args()

def set_seed(seed):
    # 设置所有随机数生成器的种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_dataset_module(dataset_name):
    """根据数据集名称获取对应的模块"""
    if dataset_name == 'cifar10':
        return cifar10, 10
    elif dataset_name == 'cifar100':
        return cifar100, 100
    elif dataset_name == 'mnist':
        return mnist, 10
    elif dataset_name == 'femnist':
        return femnist, 62  # FEMNIST数据集有62个类别（10个数字、26个大写字母和26个小写字母）
    elif dataset_name == 'fashion':
        return fashion, 10  # Fashion-MNIST有10个类别
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

def get_model(dataset_name):
    """根据数据集获取适合的模型"""
    if dataset_name =='cifar10':
        return CIFAR10ResNet18
    elif dataset_name == 'cifar100':
        return CIFAR100ResNet18
    elif dataset_name == 'femnist':
        from models.femnist_model import FEMNISTModel
        return FEMNISTModel
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

def main():
    args = parse_args()

    # 确保任务目录存在
    os.makedirs(args.task_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)

    # 设置数据路径
    flgo.set_data_root(args.data_root)

    # 获取数据集模块和类别数
    dataset_module, num_classes = get_dataset_module(args.dataset)

    # 确定算法和算法名称
    algo_name = args.algorithm.lower()
    if args.algorithm == 'FLAtten':
        algorithm = flatten
    elif args.algorithm == 'FedProx':
        algorithm = fedprox
    elif args.algorithm == 'Scaffold':
        algorithm = scaffold
    elif args.algorithm == 'FedNova':
        algorithm = fednova
    elif args.algorithm == 'MOON':
        algorithm = moon
    elif args.algorithm == 'FedDyn':
        algorithm = feddyn
    elif args.algorithm == 'Ditto':
        algorithm = ditto
    else:
        algorithm = fedavg

    # 构建任务名称，包含算法名
    task_name = f"{algo_name}_{args.dataset}_c{args.num_clients}_d{args.alpha}_r{args.num_rounds}_epoch{args.num_epochs}"


    # 组合任务目录和任务名称
    full_task_name = os.path.join(args.task_dir, task_name)

    # 获取合适的模型
    model = get_model(args.dataset)

    # 生成任务
    if args.gen_task:
        print(f"正在生成任务: {full_task_name}")
        flgo.gen_task_by_(
            dataset_module,
            fbp.DirichletPartitioner(alpha=args.alpha, num_clients=args.num_clients),
            full_task_name,
            overwrite=True
        )

    # 初始化运行器时加入新参数
    runner = flgo.init(
        full_task_name,
        algorithm,
        {
            'local_test': True, 
            'gpu': [args.gpu],
            'num_rounds': args.num_rounds,
            'num_epochs': args.num_epochs,
            'num_clients': args.num_clients,
            'proportion': args.proportion,
            'seed': args.seed,
            'dataseed': args.seed,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'optimizer': args.my_optimizer,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'clip_grad': args.clip_grad,
            'sample': args.sample,
            'aggregate': args.aggregate,
            'early_stop': args.early_stop,
            'learning_rate_decay': args.learning_rate_decay,
            'lr_scheduler': args.lr_scheduler,
            'train_holdout': args.train_holdout,
            'test_holdout': args.test_holdout,
            'eval_interval': args.eval_interval,
            'log_level': args.log_level,
        },
        model=model,
        Logger=ClassicalLogger,
    )

    # 启动训练
    if args.run:
        print(f"开始训练任务: {full_task_name}")
        runner.run()

    # 结果分析
    if args.analyze:
        print(f"分析任务结果: {full_task_name}")
        analyze_results(full_task_name, algorithm, task_name)


def analyze_results(task_name, algo, display_name):
    import flgo.experiment.analyzer as al

    # 获取算法名称，处理模块或类的情况
    algo_name = algo.__name__.split(".")[-1]
    print(algo_name)
    # 定义分析计划
    analysis_plan = {
        'Selector': {
            'task': task_name,
            'header': [algo_name, ]  # 使用处理后的算法名称
        },
        'Painter': {
            'Curve': [
                {'args': {'x': 'communication_round', 'y': 'val_client_mean_accuracy'},
                 'fig_option': {'title': f'{display_name} - Client Average Validation Accuracy'}},
                {'args': {'x': 'communication_round', 'y': 'test_global_server_accuracy'},
                 'fig_option': {'title': f'{display_name} - Global Test Accuracy'}},
            ]
        }
    }

    # 显示分析结果
    flgo.experiment.analyzer.show(analysis_plan)

    # 获取记录数据，使用处理后的算法名称
    records = al.Selector({'task': task_name, 'header': [algo_name, ]}).records[task_name]

    for rec in records:
        client_datavol = rec.data['client_datavol']
        client_metrics_all = rec.data['client_metrics_all']
        test_global_server_accuracy = rec.data['test_global_server_accuracy']
        val_client_mean_accuracy = rec.data['val_client_mean_accuracy']

        max_test_acc = max(test_global_server_accuracy)
        max_val_acc = max(val_client_mean_accuracy)

        print(f"任务: {display_name}")
        print("客户端数据量:", client_datavol)
        print("全局测试准确率:", test_global_server_accuracy)
        print(f"最大全局测试准确率 ({display_name}): {max_test_acc:.4f}")
        print("客户端平均验证准确率:", val_client_mean_accuracy)
        print(f"最大客户端平均验证准确率 ({display_name}): {max_val_acc:.4f}")

        # 绘制每个客户端的准确率曲线
        plot_client_accuracies(client_metrics_all, display_name)


def plot_client_accuracies(client_metrics_all, task_name):
    # 计算轮次
    num_rounds = len(client_metrics_all)
    num_clients = len(client_metrics_all[0]["accuracy"])
    rounds = np.arange(1, num_rounds + 1)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    for client_idx in range(num_clients):
        accuracies = [round_data["accuracy"][client_idx] for round_data in client_metrics_all]
        plt.plot(rounds, accuracies, label=f'Client {client_idx + 1}')

    # 添加标签和标题
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title(f'{task_name} - Client Accuracy per Round')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()