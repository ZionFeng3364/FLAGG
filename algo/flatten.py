import collections
import copy, torch
from flgo.algorithm import fedbase
from flgo.utils import fmodule
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader


class ExpertMLP(nn.Module): # <-- 继承自 nn.Module
    def __init__(self, dim_in=512, dim_hidden=128, dim_out=10):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, dim_in, num_experts):
        super().__init__()
        # 可以设计一个简单的两层网络
        self.layer = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x):
        # 输出原始的 logits，softmax 将在训练/推理逻辑中应用
        return self.layer(x)

class Server(fedbase.BasicServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_experts = {}


    def initialize(self, *args, **kwargs):
        """
        在服务器初始化时，将原始的 test_data 划分为新的 val_data 和 test_data。
        """
        # 检查原始的 test_data 是否存在且不为空
        original_test_data = getattr(self, 'test_data', None)
        if not original_test_data:
            self.gv.logger.warning("Server has no 'test_data' to split for validation.")
            return

        # 定义划分比例，例如 50% 作为验证集，50% 作为新的测试集
        # 这是一个可以调整的超参数
        split_ratio = 0.5
        val_size = int(len(original_test_data) * split_ratio)
        test_size = len(original_test_data) - val_size

        # 使用固定的随机种子，确保每次运行的划分结果都一样，便于复现
        generator = torch.Generator().manual_seed(42)

        # 使用 PyTorch 的 random_split 函数进行划分
        try:
            self.val_data, self.moe_test_data = random_split(
                original_test_data,
                [val_size, test_size],
                generator=generator
            )
            self.gv.logger.info(f"Server's original test data (size: {len(original_test_data)}) has been split.")
            self.gv.logger.info(f"New server validation set ('val_data') size: {len(self.val_data)}")
            self.gv.logger.info(f"New server test set ('moe_test_data') size: {len(self.moe_test_data)}")
        except Exception as e:
            self.gv.logger.error(f"Failed to split server's test data: {e}")
            # 如果划分失败，保留原始的 test_data
            self.moe_test_data = original_test_data

        # ====================  关 键 修 改 (1/3)  ====================
        # 在初始化时就创建好固定大小的门控网络

        # 假设总客户端数在此时已经可用
        self.num_total_clients = len(self.clients)

        # 假设 ResNet18 的特征维度是 512，如果你的模型不同，需要修改这里
        feature_dim = 512

        # 创建一个一次性的、足够大的门控网络
        self.gating_network = GatingNetwork(feature_dim, self.num_total_clients)

        # 创建优化器
        self.gating_lr = self.option.get('gating_lr', 0.001)
        self.gating_optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=self.gating_lr)

        self.gating_epochs = self.option.get('gating_epochs', 5)
        self.gv.logger.info(
            f"Initialized a fixed-size Gating Network for {self.num_total_clients} total possible experts.")
        # =============================================================
        return

    def iterate(self):
        """
        修改后的迭代流程，用于接收和存储专家模型。
        """
        # 1. 采样客户端 (无变化)
        self.selected_clients = self.sample()
        if not self.selected_clients:
            return False

        # 2. 与客户端通信，接收完整的包
        packages_received = self.communicate(self.selected_clients)

        # 从收到的包中解压出主模型、专家模型和客户端ID
        main_models = packages_received.get('model', [])
        expert_models = packages_received.get('distilled_expert', [])

        # flgo 的通信机制会自动添加 '__cid' 键，包含发送方客户端的ID列表
        client_ids = packages_received.get('__cid', [])
        # 3. 存储/更新客户端的专家模型 (新逻辑)
        if expert_models and client_ids:
            for cid, expert in zip(client_ids, expert_models):
                # 将专家模型存储在字典中，如果已存在则覆盖
                # 将模型移动到服务器的设备上 (如GPU)，以备将来使用
                self.client_experts[cid] = expert.to(self.device)
                # (可选) 打印日志，确认收到
                # self.gv.logger.info(f"Server: Received and stored expert from Client {cid}.")

        # 如果没有收到任何模型，本轮可能无效，提前返回
        if not main_models:
            return False

        self.model = self.aggregate(main_models)

        # 2. 训练门控网络
        self.train_gating_network()

        # 3. 评估 MoE 专家系统 (调用我们新写的独立方法)
        self.gv.logger.info("------ Evaluating MoE Expert System ------")
        self.test_moe_system()

        self.gv.logger.info(f"==========================================================")

        return True  # 表示本轮迭代成功

    def train_gating_network(self):
        """
        训练门控网络。现在它处理固定大小的输出和动态的活跃专家。
        """
        # 获取当前轮次可用的专家
        active_expert_cids = sorted(list(self.client_experts.keys()))
        if len(active_expert_cids) < 2 or not self.val_data:
            self.gv.logger.info("Skipping gating network training: Not enough experts or no validation data.")
            return

        self.gv.logger.info(f"Training Gating Network with {len(active_expert_cids)} active experts...")

        # 准备模型
        gating_network = self.gating_network.to(self.device)
        gating_network.train()
        feature_extractor = self.model.feature_extractor.to(self.device)
        feature_extractor.eval()
        active_experts = {cid: self.client_experts[cid].to(self.device) for cid in active_expert_cids}

        dataloader = DataLoader(self.val_data, batch_size=self.option['test_batch_size'], shuffle=True)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.gating_epochs):
            for batch_data in dataloader:
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data['data'], batch_data['label']
                else:
                    inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.gating_optimizer.zero_grad()

                with torch.no_grad():
                    features = torch.flatten(feature_extractor(inputs), 1)

                # ====================  关 键 修 改 (2/3)  ====================
                # 1. 门控网络预测所有可能专家的权重 (输出维度是固定的)
                all_gating_logits = gating_network(features)

                # 2. 只抽取出当前 "活跃" 专家的logits
                # active_expert_cids 就是我们需要的索引！
                active_gating_logits = all_gating_logits[:, active_expert_cids]

                # 3. 在活跃专家的logits上应用softmax，得到它们的相对权重
                active_gating_weights = F.softmax(active_gating_logits, dim=1).unsqueeze(-1)
                # =============================================================

                # 4. 获取活跃专家的预测
                # 确保顺序与 active_expert_cids 一致
                expert_outputs = torch.stack([active_experts[cid](features) for cid in active_expert_cids])
                expert_outputs = expert_outputs.permute(1, 0, 2)

                # 5. 加权求和并计算损失
                final_output = (active_gating_weights * expert_outputs).sum(dim=1)
                loss = criterion(final_output, labels)
                loss.backward()
                self.gating_optimizer.step()

        gating_network.to('cpu')
        for expert in active_experts.values():
            expert.to('cpu')

    def test_moe_system(self):
        """
        评估MoE系统，逻辑与训练时类似。
        """
        active_expert_cids = sorted(list(self.client_experts.keys()))
        if self.gating_network is None or not self.moe_test_data or len(active_expert_cids) < 1:
            self.gv.logger.info("Skipping MoE system evaluation: Components are not ready.")
            return {}

        gating_network = self.gating_network.to(self.device)
        gating_network.eval()
        feature_extractor = self.model.feature_extractor.to(self.device)
        feature_extractor.eval()
        active_experts = {cid: self.client_experts[cid].to(self.device) for cid in active_expert_cids}

        dataloader = DataLoader(self.moe_test_data, batch_size=self.option['test_batch_size'])
        total_loss, num_correct, total_samples = 0, 0, 0
        criterion = nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data['data'], batch_data['label']
                else:
                    inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = torch.flatten(feature_extractor(inputs), 1)

                # ====================  关 键 修 改 (3/3)  ====================
                # 评估时也遵循同样的逻辑
                all_gating_logits = gating_network(features)
                active_gating_logits = all_gating_logits[:, active_expert_cids]
                active_gating_weights = F.softmax(active_gating_logits, dim=1).unsqueeze(-1)
                # =============================================================

                expert_outputs = torch.stack([active_experts[cid](features) for cid in active_expert_cids])
                expert_outputs = expert_outputs.permute(1, 0, 2)

                final_output = (active_gating_weights * expert_outputs).sum(dim=1)

                total_loss += criterion(final_output, labels).item()
                _, preds = torch.max(final_output, 1)
                num_correct += (preds == labels).sum().item()
                total_samples += len(labels)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = num_correct / total_samples if total_samples > 0 else 0
        metrics = {'test_moe_accuracy': accuracy, 'test_moe_loss': avg_loss}
        self.gv.logger.info(f"MoE System Evaluation on MoE Test Set: Accuracy = {accuracy:.4f}, Loss = {avg_loss:.4f}")

        gating_network.to('cpu')
        for expert in active_experts.values():
            expert.to('cpu')
        return metrics

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
        # 为了避免传输大量模型到GPU，我们确保所有要打包的模型都在CPU上
        gating_network_cpu = self.gating_network.to('cpu')
        experts_cpu = {cid: expert.to('cpu') for cid, expert in self.client_experts.items()}

        return {
            "model": copy.deepcopy(self.model),
            "current_round": self.current_round,
            "gating_network": copy.deepcopy(gating_network_cpu),
            "client_experts": copy.deepcopy(experts_cpu)
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
        r"""
        Aggregate the locally trained models into the new one. The aggregation
        will be according to self.aggregate_option where

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==========================================================================================================================
        N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k


        Args:
            models (list): a list of local models

        Returns:
            the aggregated model

        Example:
        ```python
            >>> models = [m1, m2] # m1, m2 are models with the same architecture
            >>> m_new = self.aggregate(models)
        ```
        """
        if len(models) == 0: return self.model
        nan_exists = [m.has_nan() for m in models]
        if any(nan_exists):
            if all(nan_exists): raise ValueError("All the received local models have parameters of nan value.")
            self.gv.logger.info(
                'Warning("There exists nan-value in local models, which will be automatically removed from the aggregatino list.")')
            new_models = []
            received_clients = []
            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0 - sum(p)) * self.model + w
        else:
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            sump = sum(p)
            p = [pk / sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

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
        # 用于保存每一轮蒸馏出的专家模型 (nn.Module 类型)
        self.distilled_expert = None

        # 从配置中获取蒸馏相关的超参数
        self.distill_T = self.option.get('distill_T', 2.0)
        self.distill_alpha = self.option.get('distill_alpha', 0.5)
        self.distill_epochs = self.option.get('distill_epochs', 2)
        # 为专家模型蒸馏单独指定学习率
        self.distill_lr = self.option.get('distill_lr', 0.01)

        # 用于暂存从服务器接收的MoE系统组件
        self.server_gating_network = None
        self.server_client_experts = None

    @fmodule.with_multi_gpus
    def train(self, model):
        """
        本地计算的核心。
        1. 正常训练从服务器接收的主模型 (FModule)。
        2. 训练完成后，以此模型为教师，蒸馏出一个新的专家模型 (nn.Module)。
        """
        # ================== 阶段 A: 正常训练主模型 (FModule) ==================
        # 这一部分完全使用 flgo 的标准流程
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum
        )

        for _ in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

        # ================== 阶段 B: 蒸馏出专家模型 (nn.Module) ==================
        # 这一部分完全使用原生的 PyTorch 流程，与 flgo 解耦
        teacher_model = model
        teacher_model.eval()

        # 1. 创建一个新的专家模型 (nn.Module)
        expert_mlp = ExpertMLP(dim_in=512, dim_out=10).to(self.device)
        expert_mlp.train()

        # 2. 创建原生的优化器和损失函数
        optimizer_expert = torch.optim.SGD(expert_mlp.parameters(), lr=self.distill_lr)
        criterion_ce = nn.CrossEntropyLoss()

        # 3. 进行多轮蒸馏训练
        distill_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.loader_num_workers
        )

        for _ in range(self.distill_epochs):
            for batch_data in distill_loader:
                # flgo 的 get_batch_data 可能返回一个字典，我们需要解包
                # 假设它返回的是 (inputs, labels)
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data['data'], batch_data['label']
                else:
                    inputs, labels = batch_data

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer_expert.zero_grad()

                # a. 获取教师的软标签和特征
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                    features = teacher_model.feature_extractor(inputs)
                    features = torch.flatten(features, 1)

                # b. 获取专家的预测
                expert_logits = expert_mlp(features)

                # c. 计算总损失 (使用原生 PyTorch 函数)
                loss_ce = criterion_ce(expert_logits, labels)

                loss_kd = F.kl_div(
                    F.log_softmax(expert_logits / self.distill_T, dim=1),
                    F.softmax(teacher_logits / self.distill_T, dim=1),
                    reduction='batchmean'
                )

                total_loss = (1 - self.distill_alpha) * loss_ce + self.distill_alpha * (self.distill_T ** 2) * loss_kd

                total_loss.backward()
                optimizer_expert.step()

        # 4. 保存蒸馏出的专家模型作为“副产品”
        self.distilled_expert = expert_mlp.to('cpu')  # 存到 CPU 以节省显存

        # 使用【刚刚训练完的】主模型作为特征提取器，这是最准确的评估方式
        # self.test_distilled_expert(teacher_for_feature_extraction=teacher_model)

        # 检查是否从服务器收到了MoE系统组件
        if self.server_gating_network and self.server_client_experts:
            self.test_server_moe_on_local_data(model)

        # 5. train 方法的最终返回值必须是训练好的【主模型 FModule】
        return model

    def test_distilled_expert(self, teacher_for_feature_extraction):
        """
        一个专门用于测试本地最新蒸馏出的专家模型的函数。
        """
        # 1. 检查专家模型和测试数据是否存在
        if self.distilled_expert is None:
            # print(f"Client {self.id}: No expert model to test.")
            return

        # flgo 中，客户端通常没有 self.test_data，而是有 self.val_data
        # 我们优先使用 val_data，如果不存在，则尝试 test_data
        if hasattr(self, 'test_data') and self.test_data:
            dataset = self.test_data
        elif hasattr(self, 'val_data') and self.val_data:
            dataset = self.val_data
        else:
            print(f"Client {self.id}: No validation/test data for expert testing.")
            return

        # 2. 准备模型和数据
        expert_to_test = self.distilled_expert.to(self.device)
        expert_to_test.eval()

        feature_extractor = teacher_for_feature_extraction.feature_extractor.to(self.device)
        feature_extractor.eval()

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.test_batch_size)

        # 3. 执行测试循环
        num_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data['data'], batch_data['label']
                else:
                    inputs, labels = batch_data

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = feature_extractor(inputs)
                features = torch.flatten(features, 1)
                logits = expert_to_test(features)

                _, preds = torch.max(logits, 1)
                num_correct += (preds == labels).sum().item()
                total_samples += len(labels)

        accuracy = num_correct / total_samples if total_samples > 0 else 0.0

        # 4. 打印结果
        # 使用 flgo 的 logger 来记录，这样结果会和主模型的测试结果一起出现在日志文件中
        # logger 通常在 self.gv (global variables) 中
        log_msg = f"Round {self.current_round}, Client {self.id}: Distilled Expert Accuracy = {accuracy:.4f}"
        if hasattr(self, 'gv') and hasattr(self.gv, 'logger'):
            self.gv.logger.info(log_msg)
        else:
            print(log_msg)

        # 将专家模型移回 CPU 以节省显存
        self.distilled_expert.to('cpu')
    
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
            "distilled_expert": self.distilled_expert
        }

    def unpack(self, received_pkg):
        """
        解包服务器消息：接收全局模型，并将其 head 替换为自己的私有 head。
        """
        global_model = received_pkg['model']
        self.server_gating_network = received_pkg.get('gating_network')
        self.server_client_experts = received_pkg.get('client_experts')

        # 如果是第一轮，客户端还没有自己的私有 head，
        # 就从服务器下发的模型中复制一个作为初始化的 head。
        if self.head is None:
            self.head = copy.deepcopy(global_model.head)

        # 核心操作：用自己的私有 head 替换全局模型的 head
        model_to_train = global_model
        model_to_train.head = self.head

        # 将组合后的模型传递给后续的 train 方法
        return model_to_train

    def test_server_moe_on_local_data(self, main_model_for_feature_extraction):
        """
        在客户端自己的测试/验证数据上，评估从服务器下发的MoE系统。
        """
        # 1. 确定使用哪个数据集进行测试
        # 优先使用客户端的验证集 `val_data`，如果不存在，则使用测试集 `test_data`
        if hasattr(self, 'test_data') and self.test_data:
            dataset = self.test_data
        elif hasattr(self, 'val_data') and self.val_data:
            dataset = self.val_data
        else:
            self.gv.logger.warning(f"Client {self.id}: No local data for MoE system evaluation.")
            return

        # 2. 准备模型和数据
        # 将所有需要的模型移到客户端的设备上
        gating_network = self.server_gating_network.to(self.device)
        gating_network.eval()

        # 从主模型中获取最新的特征提取器
        feature_extractor = main_model_for_feature_extraction.feature_extractor.to(self.device)
        feature_extractor.eval()

        # 准备所有专家模型
        active_experts = {cid: expert.to(self.device) for cid, expert in self.server_client_experts.items()}
        for expert in active_experts.values():
            expert.eval()

        # 注意：这里的 active_expert_cids 必须与服务器端训练时使用的顺序一致
        # 我们假设服务器端是用 sorted keys 训练的
        active_expert_cids = sorted(list(active_experts.keys()))

        dataloader = DataLoader(dataset, batch_size=self.test_batch_size)
        total_loss, num_correct, total_samples = 0, 0, 0
        criterion = nn.CrossEntropyLoss(reduction='sum')

        # 3. 执行评估循环
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data['data'], batch_data['label']
                else:
                    inputs, labels = batch_data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = torch.flatten(feature_extractor(inputs), 1)

                all_gating_logits = gating_network(features)
                active_gating_logits = all_gating_logits[:, active_expert_cids]
                active_gating_weights = F.softmax(active_gating_logits, dim=1).unsqueeze(-1)

                expert_outputs = torch.stack([active_experts[cid](features) for cid in active_expert_cids])
                expert_outputs = expert_outputs.permute(1, 0, 2)

                final_output = (active_gating_weights * expert_outputs).sum(dim=1)

                total_loss += criterion(final_output, labels).item()
                _, preds = torch.max(final_output, 1)
                num_correct += (preds == labels).sum().item()
                total_samples += len(labels)

        # 4. 计算并打印结果
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            accuracy = num_correct / total_samples

            log_msg = f"\nClient {self.id} evaluated Server's MoE System on its local data: Accuracy = {accuracy:.4f}, Loss = {avg_loss:.4f}"
            self.gv.logger.info(log_msg)

        # 5. 清理GPU资源
        gating_network.to('cpu')
        for expert in active_experts.values():
            expert.to('cpu')
