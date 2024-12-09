#---->
import torch
import torch.nn as nn
import torchmetrics

#---->
import pytorch_lightning as pl

class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, debug):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model(model)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()

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

        self.debug = debug
        if self.debug:
            self.log_grad_histogram_every = 100
            self.last_logged_epoch = -1
            self.last_logged_step_act = 0
            self.last_logged_step_hist = 0
            self.log_activations_every = 100
            self.log_weight_histogram_every = 1

            self.activations = {}
            self._register_activation_hooks()

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, pos, label = batch

        results_dict = self.model(data=data, pos=pos, label=label)

        logits = results_dict['logits']

        loss = self.loss(logits, label)

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def on_train_epoch_end(self):
        """Logs the weight histograms periodically."""

        if not self.debug:
            return

        if (self.current_epoch - self.last_logged_epoch) >= self.log_weight_histogram_every:
            for name, param in self.named_parameters():
                if 'weight' in name:  # 只记录权重，忽略偏置等其他参数
                    self.logger.experiment.add_histogram(
                        tag=f"weights/{name}",
                        values=param.detach().cpu().numpy(),
                        global_step=self.global_step
                    )
            self.last_logged_epoch = self.current_epoch

    def validation_step(self, batch, batch_idx):
        data, pos, label = batch

        with torch.no_grad():
            results_dict = self.model(data=data, pos=pos, label=label)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']

        self.valid_metrics.update(Y_prob, label)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'label' : label}

    def on_validation_epoch_end(self):


        res = self.valid_metrics.compute()

        self.log('val_acc', res['val_MulticlassAccuracy'], logger=True)
        self.log('val_acc', res['val_MulticlassAUROC'], logger=True)
        self.log('val_acc', res['val_MulticlassF1Score'], logger=True)

        if self.best_acc is None:
            self.best_acc = res
        else:
            if self.best_acc['val_MulticlassAccuracy'] < res['val_MulticlassAccuracy']:
                self.best_acc = res

        print('best performance, ACC: {}, AUC: {}, F1: {}'.format(self.best_acc['val_MulticlassAccuracy'], self.best_acc['val_MulticlassAUROC'], self.best_acc['val_MulticlassF1Score']))

        self.valid_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def test_step(self, batch, batch_idx):

        return self.validation_step(batch, batch_idx)


    def on_test_epoch_end(self):
        self.on_validation_epoch_end()


    def load_model(self, model):
        if model.lower() == 'transmil':
            from models.transmil import TransMIL
            model = TransMIL(2)

        elif model.lower() == 'mixer':
            from models.mixer import MlpMixer
            # model = MlpMixer(input_dim=1024, hidden_dim=512, scale=32768)
            model = MlpMixer(input_dim=1024, hidden_dim=512, scale=512)

        else:
            raise ValueError("wrong model name")
        self.model = model

        print(self.model)




    def on_after_backward(self):
        """Logs the global gradient norm after each backward pass."""
        # Calculate global gradient norm

        if not self.debug:
            return

        # 全局grad范数
        grad_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), 2) for p in self.parameters() if p.grad is not None
            ]),
            2
        )
        
        # Log the global gradient norm
        self.logger.experiment.add_scalar('global_grad_norm', grad_norm, self.global_step)


        # 梯度直方图
        if (self.global_step - self.last_logged_step_hist) >= self.log_grad_histogram_every:
            for name, param in self.named_parameters():
                if param.grad is not None:

                    # torch.save(param.grad.cpu(), 'test.pt')
                    # print(torch.isnan(param.grad).sum())
                    # print(torch.isinf(param.grad).sum())
                    # print(param.grad.shape, name)
                    self.logger.experiment.add_histogram(
                        tag=f"gradients/{name}",
                        values=param.grad,
                        global_step=self.global_step
                    )
            self.last_logged_step_hist = self.global_step



        # print(self.global_step, self.last_logged_step_act, self.log_activations_every)
        """在反向传播之后记录激活值直方图"""
        if (self.global_step - self.last_logged_step_act) >= self.log_activations_every:
            for name, activation in self.activations.items():
                self.logger.experiment.add_histogram(
                    tag=f"activations/{name}",
                    values=activation,
                    global_step=self.global_step
                )

            self.last_logged_step_act = self.global_step



    def _register_activation_hooks(self):
        """为每一层注册前向传播钩子"""
        def get_activation(name):
            def hook(model, input, output):
                if (self.global_step - self.last_logged_step_act) >= self.log_activations_every:
                    if isinstance(output, dict):
                        self.activations['model_output'] = output['logits'].detach().cpu()
                    else:
                        self.activations[name] = output.detach().cpu()
            return hook

        for name, layer in self.model.named_modules():
            layer.register_forward_hook(get_activation(name))
        