import collections
import copy, torch
from flgo.algorithm import fedbase
from flgo.utils import fmodule
import torch.nn as nn


class Server(fedbase.BasicServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        r"""API for customizing the initializing process of the object"""
        return

    def iterate(self):
        """
        The standard iteration of each federated communication round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.

        Returns:
            False if the global model is not updated in this iteration
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models)
        return len(models) > 0

    def pack(self, client_id, mtype=0, *args, **kwargs):
        r"""
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.

        Args:
            client_id (int): the id of the client to communicate with
            mtype: the message type

        Returns:
            a dict contains necessary information (e.g. a copy of the global model as default)
        """
        return {
            "model": copy.deepcopy(self.model),
            "current_round": self.current_round
        }
    
    def test(self, model=None, flag:str='test'):
        r"""
        Evaluate the model on the test dataset owned by the server.

        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metrics (dict): the dict contains the evaluating results
        """
        if model is None: model = self.model
        dataset = getattr(self, flag+'_data') if hasattr(self, flag+'_data') else None
        if dataset is None:
            return {}
        else:
            if self.option['server_with_cpu']: model.to('cuda')
            if self.option['test_parallel'] and torch.cuda.device_count()>1:
                test_model = nn.DataParallel(model.to('cuda'))
                self.calculator.device = torch.device('cuda')
            else:
                test_model = model
            res = self.calculator.test(test_model, dataset, batch_size=min(self.option['test_batch_size'], len(dataset)), num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])
            self.calculator.device = self.device
            model.to(self.device)
            return res
    
    def unpack(self, packages_received_from_clients):
        r"""
        Unpack the information from the received packages. Return models and losses as default.

        Args:
            packages_received_from_clients (list): a list of packages

        Returns:
            res (dict): collections.defaultdict that contains several lists of the clients' reply
        """
        if len(packages_received_from_clients) == 0: return collections.defaultdict(list)
        res = {pname: [] for pname in packages_received_from_clients[0]}
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res

    def aggregate(self, models: list, *args, **kwargs):
        """
        核心聚合逻辑：通过 state_dict 对 feature_extractor 的浮点数参数进行加权平均。
        非浮点数类型的缓冲区直接复制。
        """
        if not models:
            return self.model

        # 1. 获取所有客户端上传的 feature_extractor 的 state_dict
        client_fe_state_dicts = [m.feature_extractor.state_dict() for m in models]

        # 2. 计算聚合权重
        weights = self.clients_contribution_to_weights(models)

        # 3. 初始化一个新的 state_dict 用于存放聚合结果，以第一个客户端的为模板
        aggregated_fe_state_dict = copy.deepcopy(client_fe_state_dicts[0])

        # 4. 遍历所有参数/缓冲区
        for key in aggregated_fe_state_dict.keys():
            # 【关键修正】检查当前项的数据类型
            if aggregated_fe_state_dict[key].dtype == torch.float32 or aggregated_fe_state_dict[
                key].dtype == torch.float64:
                # 如果是浮点数类型，则进行加权聚合
                aggregated_fe_state_dict[key].zero_()
                for i in range(len(client_fe_state_dicts)):
                    aggregated_fe_state_dict[key] += client_fe_state_dicts[i][key] * weights[i]
            else:
                # 如果不是浮点数（例如 Long 类型的 num_batches_tracked），
                # 我们不进行加权平均，直接采用第一个客户端的值即可。
                # 因为 aggregated_fe_state_dict 是从第一个客户端深拷贝的，所以这里无需任何操作。
                pass

        # 5. 将聚合后的新 state_dict 加载回服务器模型的 feature_extractor
        self.model.feature_extractor.load_state_dict(aggregated_fe_state_dict)

        return self.model

    def clients_contribution_to_weights(self, models):
        """辅助函数：根据配置计算客户端的聚合权重。"""
        # 默认使用均匀权重
        weights = [1.0 / len(models)] * len(models)

        # 如果配置为数据量加权
        if self.aggregation_option != 'uniform':
            local_data_vols = [self.clients[cid].datavol for cid in self.received_clients]
            total_data_vol = sum(local_data_vols)
            if total_data_vol > 0:
                if self.aggregation_option == 'weighted_com':
                    weights = [vol / total_data_vol for vol in local_data_vols]
                else:
                    weights = [vol / sum(local_data_vols) for vol in local_data_vols]
        return weights



class Client(fedbase.BasicClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.task = self.option['task']
        self.head = None


    @fmodule.with_multi_gpus
    def train(self, model):
        r"""
        Standard local training procedure. Train the transmitted model with
        local training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
        return
    
    @fmodule.with_multi_gpus
    def test(self, model, flag='val'):
        r"""
        Evaluate the model on the dataset owned by the client

        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metric (dict): the evaluating results (e.g. metric = {'loss':1.02})
        """
        dataset = getattr(self, flag + '_data') if hasattr(self, flag + '_data') else None
        if dataset is None: return {}
        if self.option['test_parallel'] and torch.cuda.device_count() > 1:
            test_model = nn.DataParallel(model.to('cuda'))
            self.calculator.device = torch.device('cuda')
        else:
            test_model = model
        res = self.calculator.test(test_model, dataset, min(self.test_batch_size, len(dataset)), self.option['num_workers'])
        model.to(self.device)
        self.calculator.device = self.device
        return res
    
    def get_batch_data(self):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size,
                                                                   num_workers=self.loader_num_workers,
                                                                   pin_memory=self.option['pin_memory'], drop_last=self.option.get('drop_last', False))
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self._train_loader)
            batch_data = next(self.data_loader)
        # clear local DataLoader when finishing local training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def pack(self, model, *args, **kwargs):
        """
        打包本地更新：在上传前，保存更新后的私有 head。
        """
        # 核心操作：在本地训练完成后，模型中的 head 已经是更新过的了。
        # 我们需要把它深拷贝一份，保存到 self.head，以供下一轮使用。
        self.head = copy.deepcopy(model.head)

        # 将整个更新后的模型打包返回给服务器。
        # 服务器会自己处理，只取 feature_extractor 部分。
        return {
            "model": model,
        }

    def unpack(self, received_pkg):
        """
        解包服务器消息：接收全局模型，并将其 head 替换为自己的私有 head。
        """
        global_model = received_pkg['model']

        # 如果是第一轮，客户端还没有自己的私有 head，
        # 就从服务器下发的模型中复制一个作为初始化的 head。
        if self.head is None:
            self.head = copy.deepcopy(global_model.head)

        # 核心操作：用自己的私有 head 替换全局模型的 head
        model_to_train = global_model
        model_to_train.head = self.head

        # 将组合后的模型传递给后续的 train 方法
        return model_to_train
