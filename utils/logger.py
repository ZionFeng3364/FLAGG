from flgo.experiment.logger import BasicLogger
import collections
import numpy as np
import copy


class ClassicalLogger(BasicLogger):
    def initialize(self):
        """在输出output中记录各用户的本地数据量，用户使用self.participants属性访问，服务器使用self.coordinator属性访问。self.output的默认键值为空列表"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))

        self.output["client_metrics_all"] = []

    def log_once(self, *args, **kwargs):
        # 服务器（coordinator）使用test方法测试全局模型的测试集性能
        test_metric = self.coordinator.test()
        for met_name, met_val in test_metric.items():
            self.output['test_global_server_' + met_name].append(met_val)

        # 服务器（coordinator）使用global_test方法测试全局模型的用户本地测试集性能加权平均
        val_metrics = self.coordinator.global_test(flag='test')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in val_metrics.items():
            # self.output['val_global_' + met_name + '_dist'].append(met_val)
            self.output['val_client_mean_' + met_name].append(1.0 * sum(
                [client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            # self.output['mean_val_global_' + met_name].append(np.mean(met_val))
            # self.output['std_val_global_' + met_name].append(np.std(met_val))

        # local performance
        cvals = []
        for c in self.participants:
            model = c.model if (hasattr(c, 'model') and c.model is not None) else self.coordinator.model
            cvals.append(c.test(model, 'val'))
        cval_dict = {}

        if len(cvals) > 0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)):
                    cval_dict[met_name].append(cvals[cid][met_name])
                # self.output['val_mean_' + met_name].append(float(np.array(cval_dict[met_name]).mean()))
        self.output["client_metrics_all"].append(cval_dict)
        self.show_current_output()
