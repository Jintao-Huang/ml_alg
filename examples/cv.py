
try:
    from .pre import *
except ImportError:
    from pre import *

CIFAR10 = tvd.CIFAR10
RUNS_DIR = os.path.join(RUNS_DIR, "cv")
DATASETS_PATH = os.environ.get(
    "DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

_train_dataset = CIFAR10(root=DATASETS_PATH, train=True, download=True)
DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0, 1, 2))
DATA_STD = (_train_dataset.data / 255.0).std(axis=(0, 1, 2))
print(DATA_MEANS, DATA_STD)
test_transform = tvt.Compose(
    [tvt.ToTensor(), tvt.Normalize(DATA_MEANS, DATA_STD)])
train_transform = tvt.Compose(
    [
        tvt.RandomHorizontalFlip(),
        tvt.RandomResizedCrop((32, 32), scale=(
            0.6, 1.0), ratio=(0.8, 1.2)),
        tvt.ToTensor(),
        tvt.Normalize(DATA_MEANS, DATA_STD),
    ])

_train_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                         transform=train_transform, download=True)
_val_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                       transform=test_transform, download=True)  # test_transform
test_dataset = CIFAR10(root=DATASETS_PATH, train=False,
                       transform=test_transform, download=True)
libs_ml.seed_everything(42, gpu_dtm=False)
train_dataset, _ = random_split(_train_dataset, [45000, 5000])
libs_ml.seed_everything(42, gpu_dtm=False)
_, val_dataset = random_split(_val_dataset, [45000, 5000])


class MyLModule(libs_ml.LModule):
    def __init__(self, model: Module, optim: Optimizer, hparams: Optional[Dict[str, Any]] = None) -> None:
        super(MyLModule, self).__init__(model, optim, hparams)
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        # self.optim, self.model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lrs = libs_ml.WarmupCosineAnnealingLR(
            optim, **hparams["lrs_hparams"])

    def training_step(self, batch: Any) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化. 如果返回None, 则training_step内进行自定义optimizer_step.
        # 此设计用于: GAN
        x_batch, y_batch = batch
        y = self.model(x_batch)
        loss = self.loss_fn(y, y_batch)
        y_acc = y.argmax(dim=-1)
        acc = libs_ml.accuracy_score(y_acc, y_batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def optimizer_step(self) -> None:
        super(MyLModule, self).optimizer_step()
        self.log("lr0", self.lrs.get_last_lr()[0], prog_bar_mean=False)
        self.lrs.step()

    def validation_step(self, batch: Any) -> Union[Tensor, float]:
        # fit
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        x_batch, y_batch = batch
        y = self.model(x_batch)
        y_acc = y.argmax(dim=-1)
        acc = libs_ml.accuracy_score(y_acc, y_batch)
        self.log("val_acc", acc)
        return acc

    def test_step(self, batch: Any) -> None:
        # test
        x_batch, y_batch = batch
        y = self.model(x_batch)
        y_acc = y.argmax(dim=-1)
        acc = libs_ml.accuracy_score(y_acc, y_batch)
        self.log("test_acc", acc)


if __name__ == "__main__":
    libs_ml.seed_everything(42, gpu_dtm=False)
    hparams = {
        "model_name": "resnet50",
        "model_hparams": {"num_classes": 10},
        "model_pretrain_model": {"url": tvm.ResNet50_Weights.DEFAULT.url},
        "dataloader_hparams": {"batch_size_train": 32, "num_workers": 4},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 5e-5, "weight_decay": 1e-5},
        "trainer_hparams": {"max_epochs": 10, "gradient_clip_norm": 5},
        "lrs_hparams": {...}
    }
    ldm = libs_ml.LDataModule(
        train_dataset, val_dataset, test_dataset, **hparams["dataloader_hparams"])
    hparams["lrs_hparams"] = {"warmup": 500, "T_max": len(ldm.train_dataloader) * hparams["trainer_hparams"]["max_epochs"],
                              "eta_min": 1e-5}
    model = tvm.resnet50(**hparams["model_hparams"])
    state_dict = torch.hub.load_state_dict_from_url(
        **hparams["model_pretrain_model"])
    state_dict = libs_ml.remove_keys(state_dict, ["fc"])
    print(model.load_state_dict(state_dict, strict=False))
    optimizer = optim.AdamW(model.parameters(), **hparams["optim_hparams"])
    runs_dir = CHECKPOINTS_PATH
    lmodel = MyLModule(model, optimizer, hparams)
    trainer = libs_ml.Trainer(
        lmodel, True, runs_dir=runs_dir, **hparams["trainer_hparams"])
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    trainer.test(ldm.test_dataloader)
