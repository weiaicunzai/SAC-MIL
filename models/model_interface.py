#---->
import torch
import torch.nn as nn
import torchmetrics

#---->
import pytorch_lightning as pl


# mem
class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model(model)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()
        # self.log_path = kargs['log']

        self.num_classes = 2

        metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes=self.num_classes, average = 'macro', task='MULTICLASS'),
                torchmetrics.AUROC(num_classes=self.num_classes, average = 'macro', task='MULTICLASS'),
                torchmetrics.F1Score(num_classes=self.num_classes, average = 'macro', task='MULTICLASS'),
            ]
        )

        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        self.best_acc = None

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        # data, label, slide_id = batch
        data, pos, label = batch

        results_dict = self.model(data=data, pos=pos, label=label)

        logits = results_dict['logits']


        loss = self.loss(logits, label)


        # tmp = list(self.model.fc1[0].parameters())[0]
        # print(tmp)
        return {'loss': loss}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        data, pos, label = batch
        # print('val_slide', slide_id, label)

        with torch.no_grad():
            results_dict = self.model(data=data, pos=pos, label=label)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']

        self.valid_metrics.update(Y_prob, label)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'label' : label}

    def on_validation_epoch_end(self):


        res = self.valid_metrics.compute()


        self.log('val_acc', res['val_MulticlassAccuracy'])


        # self.val_probs = {}
        if self.best_acc is None:
            self.best_acc = res

        else:
            if self.best_acc['val_MulticlassAccuracy'] < res['val_MulticlassAccuracy']:
                self.best_acc = res

        print('best performance, ACC: {}, AUC: {}, F1: {}'.format(self.best_acc['val_MulticlassAccuracy'], self.best_acc['val_MulticlassAUROC'], self.best_acc['val_MulticlassF1Score']))

        self.valid_metrics.reset()

    def configure_optimizers(self):
        # optimizer = create_optimizer(self.optimizer, self.model)
        # return [optimizer]

        return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def test_step(self, batch, batch_idx):

        return self.validation_step(batch, batch_idx)


    def on_test_epoch_end(self):
        self.on_validation_epoch_end()


    def load_model(self, model):
        if model.lower() == 'transmil':
            from models.TransMIL import TransMIL
            model = TransMIL(2)

        elif model.lower() == 'mixer':
            from models.mixer import MlpMixer
            model = MlpMixer(input_dim=1024, hidden_dim=512)

        else:
            raise ValueError("wrong model name")
        self.model = model

        print(self.model)
