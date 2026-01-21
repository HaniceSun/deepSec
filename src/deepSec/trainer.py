from .dataset import *
from .models import *
from .utils import *

class Trainer:
    def __init__(self, config_file='config.yaml', model_name='SpliceAI', train_file=None, val_file=None, test_file=None,
                 metrics_file=None, lr_lambda=None, print_every_n_batches=100):
        self.config_dir = f'{resources.files("deepSec").parent}/config'
        self.config = self.load_yaml(config_file)

        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}', flush=True)
        model_class = eval(self.config['models'][model_name]['class'])
        cfg = Config(self.config['models'][model_name]['params'])
        cfg.device = self.device
        self.model = model_class(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.loss_fn = CustomLoss()
        self.print_every_n_batches = print_every_n_batches

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if train_file and os.path.exists(train_file):
            self.train_dataset = torch.load(train_file, weights_only=False)
        if val_file and os.path.exists(val_file):
            self.val_dataset = torch.load(val_file, weights_only=False)
        if test_file and os.path.exists(test_file):
            self.test_dataset = torch.load(test_file, weights_only=False)

        self.metrics_file = metrics_file
        if not self.metrics_file:
            self.metrics_file = f'{model_name}_metrics.txt'

        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_confusion = []
        self.val_confusion = []
        self.test_confusion = []

        self.learning_rates = []
        self.lr_scheduler = None
        if lr_lambda:
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_lambda[epoch])

        self.early_stopping = None
        if cfg.early_stopping:
            self.early_stopping = EarlyStopping()
        self.best_epochs = []

    def train(self, epoch):
        self.model.train()

        loss_total = 0
        for nb, (X, y) in enumerate(self.train_dataset):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss.detach().item()

            if nb % self.print_every_n_batches == 0:
                print(f'train epoch:{epoch} batch:{nb} loss:{loss.detach().item()}')

        self.save_checkpoint(epoch)
        train_loss = loss_total / len(self.train_dataset)
        self.train_loss.append(train_loss)
        self.epochs.append(epoch)

        if self.lr_scheduler:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.learning_rates.append(lr)
        print(f'train epoch:{epoch} avg loss: {" ".join([str(x) for x in self.train_loss])}')

    def validate(self, epoch, test=False):
        ds = 'val'
        dataset = self.val_dataset
        loss_list = self.val_loss
        if test:
            ds = 'test'
            dataset = self.test_dataset
            loss_list = self.test_loss
            self.epochs.append(epoch)

        self.model.eval()
        loss_total = 0
        with torch.no_grad():
            for nb, (X, y) in enumerate(dataset):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                loss_total += loss.detach().item()

                if nb % self.print_every_n_batches == 0:
                    print(f'{ds} epoch:{epoch} batch:{nb} loss:{loss.detach().item()}', flush=True)
        loss_avg = loss_total / len(dataset)
        loss_list.append(loss_avg)
        print(f'{ds} epoch:{epoch} avg loss: {" ".join([str(x) for x in loss_list])}')

    def test(self, epoch):
        self.load_checkpoint(epoch)
        self.validate(epoch, test=test)
        self.get_confusion(dataset='test')
        self.log_metrics()

    def get_confusion(self, dataset='val', labels=[0, 1]):
        self.model.eval()
        if dataset == 'val':
            ds = self.val_dataset
            cml = self.val_confusion
        elif dataset == 'train':
            ds = self.train_dataset
            cml = self.train_confusion
        elif dataset == 'test':
            ds = self.test_dataset
            cml = self.test_confusion

        y_true = []
        y_pred = []
        with torch.no_grad():
            for nb, (X, y) in enumerate(ds):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.argmax(dim=1).cpu().numpy())
        cm = confusion_matrix(y_true, y_pred, labels=labels).ravel()
        cml.append(','.join([str(x) for x in cm]))
        print(f'{dataset} confusion matrix (tn, fp, fn, tp): {cm}')

    def predict(self, epoch=None, pred_seq=[], pred_file='pred_dataset.pt', pred_file_paired=True, out_file='predicted.txt', alphabet=['N', 'A', 'C', 'G', 'T']):
        self.load_checkpoint(epoch)
        self.model.eval()
        with torch.no_grad():
            if pred_seq:
                n = -1
                for seq in pred_seq:
                    n += 1
                    seq2 = torch.tensor([[alphabet.index(x) if x in alphabet else 0 for x in seq]]).to(self.device)
                    pred1 = self.model(seq2)
                    pred1 = torch.softmax(pred1, dim=1).cpu()
                    cls1 = torch.argmax(pred1).item()
                    pred1 = pred1.squeeze().numpy().tolist()
                    df = pd.DataFrame({'index':[n], 'seq':[seq], 'G1_A':[pred1[0]], 'G1_B':[pred1[1]], 'G1_cls':[cls1]})
                    if n == 0:
                        df.to_csv(out_file, index=False, header=True, sep='\t')
                    else:
                        df.to_csv(out_file, index=False, header=False, sep='\t', mode='a')
                    print([seq] + pred1 + [cls1])

            elif os.path.exists(pred_file):
                self.pred_dataset = torch.load(pred_file)
                n = -1
                if not pred_file_paired:
                    for X0, X1 in self.pred_dataset:
                        n += 1
                        X1 = X1.to(self.device)
                        pred1 = self.model(X1)
                        pred1 = torch.softmax(pred1, dim=1).cpu()
                        cls1 = torch.argmax(pred1, dim=1)
                        df = pd.DataFrame({'index':X0, 'seq':['.']*len(X0), 'G1_A':pred1[:, 0], 'G1_B':pred1[:, 1], 'G1_cls':cls1})
                        if n == 0:
                            df.to_csv(out_file, index=False, header=True, sep='\t')
                        else:
                            df.to_csv(out_file, index=False, header=False, sep='\t', mode='a')
                else:
                    for X0, X1, X2 in self.pred_dataset:
                        n += 1
                        X1, X2 = X1.to(self.device), X2.to(self.device)
                        pred1 = self.model(X1)
                        pred2 = self.model(X2)
                        pred1 = torch.softmax(pred1, dim=1).cpu()
                        pred2 = torch.softmax(pred2, dim=1).cpu()
                        cls1 = torch.argmax(pred1, dim=1)
                        cls2 = torch.argmax(pred2, dim=1)
                        df = pd.DataFrame({'index':X0, 'seq':['.']*len(X0), 'G1_A':pred1[:, 0], 'G1_B':pred1[:, 1], 'G2_A':pred2[:, 0], 'G2_B':pred2[:, 1], 'G1_cls':cls1, 'G2_cls':cls2})
                        if n == 0:
                            df.to_csv(out_file, index=False, header=True, sep='\t')
                        else:
                            df.to_csv(out_file, index=False, header=False, sep='\t', mode='a')

    def log_metrics(self):
        cols = ['epoch', 'best_epoch', 'learning_rate',
                      'train_loss', 'val_loss', 'test_loss',
                      'train_confusion', 'val_confusion', 'test_confusion']
        df = pd.DataFrame('.', index=range(len(self.epochs)), columns=cols)

        df['epoch'] = self.epochs
        if self.train_loss:
            df['train_loss'] = self.train_loss
        if self.val_loss:
            df['val_loss'] = self.val_loss
        if self.test_loss:
            df['test_loss'] = self.test_loss

        if self.train_confusion:
            df['train_confusion'] = self.train_confusion
        if self.val_confusion:
            df['val_confusion'] = self.val_confusion
        if self.test_confusion:
            df['test_confusion'] = self.test_confusion

        if self.learning_rates:
            df['learning_rate'] = self.learning_rates
        if self.best_epochs:
            df['best_epoch'] = self.best_epochs

        if os.path.exists(self.metrics_file):
            df.tail(1).to_csv(self.metrics_file, index=False, header=False, sep='\t', mode='a')
        else:
            df.tail(1).to_csv(self.metrics_file, index=False, header=True, sep='\t')

        if self.train_loss and self.val_loss:
            plot_file = self.metrics_file.replace('.txt', '.pdf')
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(df['epoch'], df['train_loss'], label='train_loss')
            ax.plot(df['epoch'], df['val_loss'], label='val_loss')
            plt.savefig(plot_file)
            plt.close()

    def saliency_map(self, epoch=None, sal_seq=[], out_file='saliency.txt', target=1, alphabet=['N', 'A', 'C', 'G', 'T']):
        self.load_checkpoint(epoch)
        self.model.eval()
        for n, seq in enumerate(sal_seq):
            seq2 = torch.tensor([[alphabet.index(x) if x in alphabet else 0 for x in seq]])
            X = self.model.OneHot(seq2)

            X.requires_grad = True
            pred = self.model(X, from_onehot=True)
            loss = self.loss_fn(pred, torch.tensor([target]).to(self.device))
            loss.backward()

            #saliency = X.grad.abs()
            #sal = np.sum(np.multiply(saliency.squeeze().cpu().numpy(), X.squeeze().cpu().numpy()), axis=0)

            saliency = -X.grad
            sal = np.clip(np.sum(np.multiply(saliency.squeeze().cpu().numpy(), X.detach().squeeze().cpu().numpy()), axis=0), 0, None)

            df = pd.DataFrame({'index':list(range(len(seq))), 'seq':list(seq), 'saliency':list(sal)})
            out_file_n = out_file.replace('.txt', f'_seq{n}.txt')
            plot_file_n = out_file.replace('.txt', f'_seq{n}.pdf')
            df.to_csv(out_file_n, index=False, header=True, sep='\t')
            fig = plt.figure(figsize=[12, 6])
            ax = fig.add_subplot()
            ax.bar(df['index'], df['saliency'])
            ax.set_xlim(-1, len(df['index']))
            ax.set_xticks(df['index'])
            ax.set_xticklabels(df['seq'], fontsize=8)
            ax.set_ylabel('Magnitude of saliency values')
            plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.tight_layout()
            plt.savefig(plot_file_n)

    def count_parameters(self, with_lazy=True, show_details=False):
        if with_lazy:
            X,y = next(iter(self.train_dataset))
            pred = self.model.forward(X)

        L = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                pnum = parameter.numel()
                L.append([name, pnum])
        df = pd.DataFrame(L)
        df.columns = ['name', 'pnum']
        if show_details:
            print(df)
        print(f'Total Trainable Parameters: {df["pnum"].sum()}')

    def set_learning_rate(self, lr=1e-3):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, epoch):
        checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }
        out_file = f'{self.model_name}_ckpt_{epoch}.pt'
        torch.save(checkpoint, out_file) 

    def load_checkpoint(self, epoch):
        in_file = f'{self.model_name}_ckpt_{epoch}.pt'
        if os.path.exists(in_file):
            print(f'loading checkpoint from {in_file}')
            checkpoint = torch.load(in_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f'checkpoint file {in_file} not found!')

    def load_yaml(self, config_file):
        if not os.path.exists(config_file):
            config_file = self.config_dir + '/' + config_file
            print(f'using config {config_file}')
        config = {}
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f'Error loading config file {e}')

    def run(self, resume_epoch=None, start_epoch=0, end_epoch=10, validate=True):
        if resume_epoch is not None:
            self.load_checkpoint(resume_epoch)
            start_epoch = resume_epoch + 1

        for epoch in range(start_epoch, end_epoch):
            self.train(epoch)
            self.get_confusion(dataset='train')
            if validate:
                self.validate(epoch)
                self.get_confusion(dataset='val')
                if self.early_stopping:
                    self.early_stopping(self.val_loss[-1], epoch)
                    self.best_epochs.append(self.early_stopping.best_epoch)

                    if self.early_stopping.stopped:
                        print(f'Early stopped, beast epoch: {self.early_stopping.best_epoch}')
                        break

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.log_metrics()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.count_parameters()
    trainer.run()
    trainer.test()
    trainer.predict()
    trainer.saliency_map()
