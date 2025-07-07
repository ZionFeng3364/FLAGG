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
            self.gv.logger.info('Warning("There exists nan-value in local models, which will be automatically removed from the aggregatino list.")')
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



class Client(fedbase.BasicClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.task = self.option['task']

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
        r"""
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.

        Args:
            model: the locally trained model

        Returns:
            package: a dict that contains the necessary information for the server
        """
        return {
            "model": model,
        }

    def unpack(self, received_pkg):
        self.current_round = received_pkg['current_round']
        return received_pkg['model']
