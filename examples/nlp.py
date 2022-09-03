# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from pre import *
from transformers.models.bert.modeling_bert import BertForSequenceClassification
logger = logging.getLogger(__name__)

device_ids = [0]

RUNS_DIR = os.path.join(RUNS_DIR, "nlp")
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class MyLModule(libs_ml.LModule):
    def __init__(self, model: Module, optimizer: Optimizer, metrics: Dict[str, Metric],
                 loss_fn: Module, lr_s: LRScheduler, hparams: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, optimizer, metrics, "f1", hparams)
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        # self.optim, self.model
        self.loss_fn = loss_fn
        self.lr_s = lr_s

    def optimizer_step(self) -> None:
        super().optimizer_step()
        self.lr_s.step()

    def _calculate_loss_prob_pred(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        y = self.model(**batch)
        loss, logits = y["loss"], y["logits"]
        y_prob = torch.softmax(logits, 1)[:, 1]
        y_pred = logits.argmax(dim=-1)
        return loss, y_prob, y_pred

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化
        loss, _, y_pred = self._calculate_loss_prob_pred(batch)
        acc = accuracy(y_pred, batch["labels"])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        # fit
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        loss, y_prob, y_pred = self._calculate_loss_prob_pred(batch)
        for k, metric in self.metrics.items():
            if k == "auc":
                metric.update(y_prob, batch["labels"])
            elif k == "loss":
                metric.update(loss)
            else:
                metric.update(y_pred, batch["labels"])

    def test_step(self, batch: Dict[str, Tensor]) -> None:
        # test
        self.validation_step(batch)


if __name__ == "__main__":
    dataset = load_dataset("glue", "mrpc")
    model_name = "bert-base-uncased"
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
    dataset = dataset.rename_column("label", "labels")
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    #
    # libs_ml.seed_everything(42, gpu_dtm=False)
    max_epochs = 10
    batch_size = 32
    n_accumulate_grad = 4
    hparams = {
        "device_ids": device_ids,
        "model_name": model_name,
        "optim_name": "SGD",  # AdamW 容易过拟合
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4, "collate_fn": collate_fn},
        "optim_hparams": {"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},  #
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 10,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad
        },
        "lrs_hparams": {
            "warmup": 30,  # 30 * n_accumulate_grad
            "T_max": ...,
            "eta_min": 1e-3
        }
    }
    hparams["lrs_hparams"]["T_max"] = math.ceil(len(dataset["train"]) // batch_size / n_accumulate_grad) * max_epochs
    #
    ldm = libs_ml.LDataModule(
        dataset["train"], dataset["validation"], dataset["test"], **hparams["dataloader_hparams"])
    #
    model: PreTrainedModel = BertForSequenceClassification.from_pretrained(model_name)
    optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
    metrics: Dict[str, Metric] = {
        "loss": MeanMetric(),
        "acc":  Accuracy(),
        "auc": AUROC(),  # 必须是二分类
        "prec": Precision(average="macro", num_classes=2),
        "recall": Recall(average="macro", num_classes=2),
        "f1": F1Score(average="macro", num_classes=2)
    }
    runs_dir = CHECKPOINTS_PATH
    loss_fn = nn.CrossEntropyLoss()
    lr_s = libs_ml.WarmupCosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
    lmodel = MyLModule(model, optimizer, metrics, loss_fn, lr_s, hparams)
    trainer = libs_ml.Trainer(lmodel, device_ids, runs_dir=runs_dir, **hparams["trainer_hparams"])
    try:
        logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    except KeyboardInterrupt:
        # nohup下, 使用`kill -2 <pid>` 产生KeyboardInterrupt
        logger.info("KeyboardInterrupt Detected...")
        raise
    finally:
        logger.info(trainer.test(ldm.test_dataloader))
