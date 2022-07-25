
try:
    from .pre import *
except ImportError:
    from pre import *

CIFAR10 = tvd.CIFAR10
act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

def setup():
    global DATASETS_PATH, CHECKPOINTS_PATH, device
    RUNS_DIR = os.path.join(PL_RUNS_DIR, "_4")
    DATASETS_PATH = os.environ.get(
        "DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
    CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")

    #

    libs_ml.seed_everything(42, gpu_dtm=False)
    device = torch.device(
        "cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)
    #
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
    # Files to download
    pretrained_files = [
        "GoogleNet.ckpt",
        "ResNet.ckpt",
        "ResNetPreAct.ckpt",
        "DenseNet.ckpt",
        "tensorboards/GoogleNet/events.out.tfevents.googlenet",
        "tensorboards/ResNet/events.out.tfevents.resnet",
        "tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact",
        "tensorboards/DenseNet/events.out.tfevents.densenet",
    ]

    # libs_utils.download_files(base_url, pretrained_files, CHECKPOINTS_PATH)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    #
    train_dataset = CIFAR10(root=DATASETS_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    print(DATA_MEANS, DATA_STD)
    test_transform = tvt.Compose(
        [tvt.ToTensor(), tvt.Normalize(DATA_MEANS, DATA_STD)])
    train_transform = tvt.Compose(
        [
            tvt.RandomHorizontalFlip(),
            tvt.RandomResizedCrop((32, 32), scale=(
                0.8, 1.0), ratio=(0.9, 1.1)),
            tvt.ToTensor(),
            tvt.Normalize(DATA_MEANS, DATA_STD),
        ])
    #
    train_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                            transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                          transform=test_transform, download=True)
    #
    pl.seed_everything(42)
    train_set, _ = random_split(train_dataset, [45000, 5000])
    pl.seed_everything(42)
    _, val_set = random_split(val_dataset, [45000, 5000])
    #
    test_set = CIFAR10(root=DATASETS_PATH, train=False,
                       transform=test_transform, download=True)
    #
    global ldm
    ldm = libs_ml.LDataModule(
        train_set, val_set, test_set, batch_size_train=128, num_workers=4)
    #
    imgs, _ = next(iter(ldm.train_dataloader))
    print("Batch mean", imgs.mean(dim=[0, 2, 3]))
    print("Batch std", imgs.std(dim=[0, 2, 3]))
    # Batch mean tensor([0.0231, 0.0006, 0.0005])
    # Batch std tensor([0.9865, 0.9849, 0.9868])


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        # c_red: reduce
        super(InceptionBlock, self).__init__()

        self.conv_1x1 = nn.Sequential(  # ConvBNReLU
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), nn.BatchNorm2d(
                c_out["1x1"]), act_fn()
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn(),
        )

        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn(),
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, act_fn_name="relu"):
        super(GoogleNet, self).__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes, act_fn_name=act_fn_name, act_fn=act_fn_by_name[act_fn_name]
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        #
        self.input_net = nn.Sequential(
            # 可以bias=False
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(
                64), self.hparams.act_fn()
        )
        #
        self.inception_blocks = nn.Sequential(
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
        )
        #
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(
                128, self.hparams.num_classes)
        )

    def _init_params(self):
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x


class MyLModule(libs_ml.LModule):
    def __init__(self, model: Module, optim: Optimizer, hparams: Optional[Dict[str, Any]] = None) -> None:
        super(MyLModule, self).__init__(model, optim, hparams)
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        # self.optim, self.model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lrs = lrs.MultiStepLR(optim, milestones=[100, 150], gamma=0.1)
        


    def epoch_end(self) -> None:
        # fit. 用于lr_schedules的处理
        # 这里log的信息不会出现在prog_bar中. 其他函数(b, o, t, v, t)中log的都会出现
        self.log("lr0", self.lrs.get_last_lr()[0])
        self.lrs.step()


    def training_step(self, batch: Any) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化. 如果返回None, 则training_step内进行自定义optimizer_step.
        # 此设计用于: GAN
        x_batch, y_batch = batch
        y = self.model(x_batch)
        loss = self.loss_fn(y, y_batch)
        y_acc = y.argmax(dim = -1)
        acc = libs_ml.accuracy_score(y_acc, y_batch)
        self.log("train_loss", loss)
        self.log("train_val", acc)
        return loss

    def validation_step(self, batch: Any) -> Union[Tensor, float]:
        # fit
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        x_batch, y_batch = batch
        y = self.model(x_batch)
        y_acc = y.argmax(dim = -1)
        acc = libs_ml.accuracy_score(y_acc, y_batch)
        self.log("val_acc", acc)
        return acc

    def test_step(self, batch: Any) -> None:
        # test
        x_batch, y_batch = batch
        y = self.model(x_batch)
        y_acc = y.argmax(dim = -1)
        acc = libs_ml.accuracy_score(y_acc, y_batch)
        self.log("test_acc", acc)


def inception():
    hparams = {
        "model_name": "GoogleNet",
        "model_hparams": {"num_classes": 10, "act_fn_name": "relu"},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 1e-3, "weight_decay": 1e-4}
    }

    model = GoogleNet(**hparams["model_hparams"])
    optimizer = optim.AdamW(model.parameters(), **hparams["optim_hparams"])
    default_root_dir = CHECKPOINTS_PATH
    lmodel = MyLModule(model, optimizer, hparams)
    trainer = libs_ml.Trainer(lmodel, True, 180, default_root_dir)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    trainer.test(ldm.val_dataloader)
    trainer.test(ldm.test_dataloader)
    # getattr(optim, "AdamW")


def resnet():
    pass


def densenet():
    pass


if __name__ == "__main__":
    # print(getattr(optim, "AdamW"))
    setup()
    inception()
