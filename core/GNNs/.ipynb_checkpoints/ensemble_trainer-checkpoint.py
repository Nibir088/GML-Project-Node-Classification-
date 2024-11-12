import torch

from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.data_utils.load import load_data

LOG_FREQ = 10


class EnsembleTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers

        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = cfg.gnn.train.feature_type
        self.epochs = cfg.gnn.train.epochs
        self.weight_decay = cfg.gnn.train.weight_decay

        # ! Load data
        data, _ = load_data(self.dataset_name, use_dgl=False, use_text=False, seed=cfg.seed)

        data.y = data.y.squeeze()
        self.data = data.to(self.device)

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )#["acc"]

        if cfg.gnn.model.name == 'RevGAT':
            self.TRAINER = DGLGNNTrainer
        else:
            self.TRAINER = GNNTrainer

    @ torch.no_grad()
    def _evaluate(self, logits):
        val_metric = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_metric = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_metric, test_metric

    @ torch.no_grad()
    def eval(self, logits):
        val_metric, test_metric = self._evaluate(logits)
        print(f'[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_metric["acc"]:.4f}, TestAcc: {test_metric["acc"]:.4f}\n')
        print(f'ValPre: {val_metric["precision"]:.4f}, TestPre: {test_metric["precision"]:.4f}\n')
        print(f'ValRecall: {val_metric["recall"]:.4f}, TestRecall: {test_metric["recall"]:.4f}\n')
        print(f'ValF1: {val_metric["f1"]:.4f}, TestF1: {test_metric["f1"]:.4f}\n')
        print(f'ValAUC: {val_metric["auc"]:.4f}, TestAUC: {test_metric["auc"]:.4f}\n')

        # Update res dictionary
        res = {
            'val_acc': val_metric['acc'],
            'test_acc': test_metric['acc'],
            'val_precision': val_metric['precision'],
            'test_precision': test_metric['precision'],
            'val_recall': val_metric['recall'],
            'test_recall': test_metric['recall'],
            'val_f1': val_metric['f1'],
            'test_f1': test_metric['f1'],
            'val_auc': val_metric['auc'],
            'test_auc': test_metric['auc']
        }
        return res

    def train(self):
        all_pred = []
        all_acc = {}
        feature_types = self.feature_type.split('_')
        for feature_type in feature_types:
            trainer = self.TRAINER(self.cfg, feature_type)
            trainer.train()
            pred, metric = trainer.eval_and_save()
            all_pred.append(pred)
            all_acc[feature_type] = metric['val_acc']
            
        pred_ensemble = sum(all_pred)/len(all_pred)
        acc_ensemble = self.eval(pred_ensemble)
        all_acc['ensemble'] = acc_ensemble
        return all_acc
    