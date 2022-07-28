# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

try:
    from .pre import *
except ImportError:
    from pre import *


RUNS_DIR = os.path.join(RUNS_DIR, "nlp")
DATASETS_PATH = os.environ.get(
    "DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda")
print("Using device", device)

dataset = load_dataset("glue", "mrpc")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
dataset = dataset.rename_column("label", "labels")
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


class MyLModule(libs_ml.LModule):
    def __init__(self, model: Module, optimizer: Optimizer, loss_fn: Module, lr_s: LRScheduler, hparams: Optional[Dict[str, Any]] = None) -> None:
        super(MyLModule, self).__init__(model, optimizer, hparams)
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        # self.optim, self.model
        self.loss_fn = loss_fn
        self.lr_s = lr_s

    def _calculate_loss_acc(self, batch: Any) -> Tuple[Tensor, Tensor]:
        y = self.model(**batch)
        # nn.CrossEntropyLoss()(y["logits"].float(), F.one_hot(batch["labels"]).float())
        loss, logits = y["loss"], y["logits"]
        y_acc = logits.argmax(dim=-1)
        acc = libs_ml.accuracy_score(y_acc, batch["labels"])
        return loss, acc

    def training_step(self, batch: Any) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化. 如果返回None, 则training_step内进行自定义optimizer_step.
        # 此设计用于: GAN
        loss, acc = self._calculate_loss_acc(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def optimizer_step(self) -> None:
        super(MyLModule, self).optimizer_step()
        self.log("lr0", self.lr_s.get_last_lr()[0], prog_bar_mean=False)
        self.lr_s.step()

    def validation_step(self, batch: Any) -> Union[Tensor, float]:
        # fit
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        loss, acc = self._calculate_loss_acc(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return acc

    def test_step(self, batch: Any) -> None:
        # test
        _, acc = self._calculate_loss_acc(batch)
        self.log("test_acc", acc)


if __name__ == "__main__":
    libs_ml.seed_everything(42, gpu_dtm=False)
    hparams = {
        "model_name": model_name,
        "optim_name": "AdamW",
        "dataloader_hparams": {"batch_size_train": 32, "num_workers": 4, "collate_fn": collate_fn},
        "optim_hparams": {"lr": 5e-5, "weight_decay": 1e-5},  #
        "trainer_hparams": {"max_epochs": 5, "gradient_clip_norm": 5},
        "lrs_hparams": {
            "warmup": 100,
            "T_max": ..., 
            "eta_min": 1e-5
        }
    }
    ldm = libs_ml.LDataModule(
        dataset["train"], dataset["validation"], dataset["test"], **hparams["dataloader_hparams"])
    hparams["lrs_hparams"]["T_max"] = len(
        ldm.train_dataloader) * hparams["trainer_hparams"]["max_epochs"]
    #
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    optimizer = getattr(optim, hparams["optim_name"])(
        model.parameters(), **hparams["optim_hparams"])
    runs_dir = CHECKPOINTS_PATH
    loss_fn = nn.CrossEntropyLoss()
    lr_s = libs_ml.WarmupCosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
    lmodel = MyLModule(model, optimizer, loss_fn, lr_s, hparams)
    trainer = libs_ml.Trainer(
        lmodel, device, runs_dir=runs_dir, **hparams["trainer_hparams"])
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    trainer.test(ldm.test_dataloader)