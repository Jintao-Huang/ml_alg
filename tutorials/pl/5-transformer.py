
try:
    from .pre import *
except ImportError:
    from pre import *
    

RUNS_DIR = os.path.join(PL_RUNS_DIR, "_5")
DATASETS_PATH = os.environ.get(
    "DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")

libs_ml.seed_everything(42)

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
#

# [The Transformer architecture]


# [What is attention]
# [Scaled dot product attention]
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        # mask == 0的地方 mask
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


# seq_len, d_k = 3, 2
# pl.seed_everything(42)
# q = torch.randn(seq_len, d_k)
# k = torch.randn(seq_len, d_k)
# v = torch.randn(seq_len, d_k)
# values, attention = scaled_dot_product(q, k, v)
# print("Q\n", q)
# print("K\n", k)
# print("V\n", v)
# print("Values\n", values)
# print("Attention\n", attention)

# [Multi-head attention]


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length,
                          self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        #
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

# [Transformer encoder]


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.):
        super(EncoderBlock, self).__init__()

        #
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),  # ReLU的Dropout前后都能放, 其他一般放后面
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        #
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        #
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

# [PE]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) / d_model * (-math.log(10000.0)))  # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# encod_block = PositionalEncoding(d_model=512, max_len=512)
# pe = encod_block.pe.squeeze().T.cpu().numpy()  # .t()

# #
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
# pos = ax.imshow(pe, origin="lower", extent=(0, pe.shape[1], 0, pe.shape[0]))
# fig.colorbar(pos, ax=ax)
# ax.set_xlabel("Position in sequence")
# ax.set_ylabel("Hidden dimension")
# ax.set_title("Positional encoding over hidden dimensions")
# ax.set_xticks([i * 50 for i in range(0, pe.shape[1] // 50)])
# ax.tick_params(axis='x', which='major', rotation=90)
# ax.set_yticks([i * 50 for i in range(0, pe.shape[0] // 50)])
# plt.show()
# plt.close()


# sns.set_theme()
# fig, ax = plt.subplots(2, 2, figsize=(12, 4))
# ax = [a for a_list in ax for a in a_list]
# for i in range(len(ax)):
#     ax[i].plot(np.arange(16), pe[i, :16], color="C%i" %
#                i, marker="o", markersize=6, markeredgecolor="black")
#     ax[i].set_title("Encoding in hidden dimension %i" % i)
#     ax[i].set_xlabel("Position in sequence", fontsize=10)
#     ax[i].set_ylabel("Positional encoding", fontsize=10)
#     ax[i].set_xticks(np.arange(16))
#     ax[i].tick_params(axis="both", which="major", labelsize=10)
#     ax[i].tick_params(axis="both", which="minor", labelsize=8)
#     ax[i].set_ylim(-1.2, 1.2)
# fig.subplots_adjust(hspace=0.8)
# sns.reset_orig()
# plt.show()

# [LR warm-up]


def get_offset_func(fa: float, fb: float, ga: float, gb: float) -> Callable[[float], float]:
    """将y=[sa..sb]的曲线 -> y=[ta..tb]. 曲线的趋势不变"""
    # 存在f, g; 已知: g(x)=s(f(x)+t), 求s,a. 返回func: f->g. s,a为标量
    # 即: 通过缩放和平移, 将f->g
    # s(f(a)+t)=g(a); s(f(b)+t)=g(b)
    if fa == fb:
        raise ValueError("fa == fb")
    if ga == gb:
        return lambda x: ga
    s = (ga-gb) / (fa-fb)
    t = ga / s - fa

    def func(x):
        return s * (x + t)
    return func


def cosine_annealing_lr(epoch: int, T_max: int, eta_min: float, initial_lrs: List[float]) -> List[float]:
    if epoch == 0:
        return initial_lrs
    if epoch == T_max:
        return [eta_min] * len(initial_lrs)
    if epoch > T_max:
        raise ValueError(f"epoch: {epoch}")
    # 余弦曲线
    #   epoch=0: lr=initial_lr
    #   epoch=T_max: lr=eta_min
    # 周期为T_max * 2的cos函数: 系数=2pix/T
    res = []
    x = math.cos(math.pi * epoch / T_max)
    # 缩放[-1, 1] -> [eta_min, initial_lr]
    for initial_lr in initial_lrs:
        func = get_offset_func(-1, 1, eta_min, initial_lr)
        res.append(func(x))
    return res


class _WarmupCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        # warmup一般使用iter_idx(epoch)作为T_max进行控制
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super(_WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        lrs = cosine_annealing_lr(
            self.last_epoch, self.T_max, self.eta_min, self.base_lrs)
        scale = 1
        if self.last_epoch <= self.warmup:
            scale = self.last_epoch / self.warmup
        return [lr * scale for lr in lrs]


# p = nn.Parameter(torch.empty(4, 4))
# optimizer = optim.Adam([p], lr=1e-3)
# lr_scheduler = _WarmupCosineAnnealingLR(
#     optimizer=optimizer, warmup=100, max_iters=2000)


#
# epochs = list(range(2000))
# sns.set()
# plt.figure(figsize=(8, 3))
# # print([lr_scheduler.get_lr_factor(e) for e in epochs])
# plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
# plt.ylabel("Learning rate factor")
# plt.xlabel("Iterations (in batches)")
# plt.title("Cosine Warm-up Learning Rate Scheduler")
# plt.show()
# sns.reset_orig()


# [PyTorch Lightning Mudule]


class MyLModule(libs_ml.LModule):
    def __init__(self, model: Module, optim: Optimizer, hparams: Optional[Dict[str, Any]] = None) -> None:
        super(MyLModule, self).__init__(model, optim, hparams)
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        # self.optim, self.model在super中定义
        self.lrs = _WarmupCosineAnnealingLR(optim, **hparams["lrs_params"])


    def optimizer_step(self) -> None:
        # fit. 用于optim, lr_schedules的处理.
        self.optim.step()
        clip_grad_norm_(self.model.parameters(), **
                        self.hparams["clip_grad_norm_params"])
        self.log("lr0", self.lrs.get_last_lr()[0])
        self.lrs.step()

    def training_step(self, batch: Any) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化. 如果返回None, 则training_step内进行自定义optimizer_step.
        # 此设计用于: GAN
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch: Any) -> Union[Tensor, float]:
        # fit. no_grad环境
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        _, acc = self._calculate_loss(batch, mode="val")
        return acc

    def test_step(self, batch: Any) -> None:
        # test. no_grad环境
        _ = self._calculate_loss(batch, mode="test")

    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        inp_data = F.one_hot(
            inp_data, num_classes=self.hparams["model_params"]["num_classes"]).float()

        #
        preds = self.model(inp_data, add_positional_encoding=True)
        loss = F.cross_entropy(
            preds.view(-1, preds.size(-1)), labels.view(-1))  # [N * SeqLen, C]
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        #
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)  # on_step=True, on_epoch=False
        return loss, acc


class MyModule(Module):
    def __init__(self, hparams) -> None:
        super(MyModule, self).__init__()
        # input_dropout, input_dim, model_dim, num_layers, num_heads, dropout, num_classes
        self.hparams = hparams
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams["input_dropout"]), nn.Linear(
                self.hparams["input_dim"], self.hparams["model_dim"])
        )  # Position_wise
        self.positional_encoding = PositionalEncoding(
            d_model=self.hparams["model_dim"])
        #
        self.transformer = TransformerEncoder(
            num_layers=self.hparams["num_layers"],
            input_dim=self.hparams["model_dim"],
            dim_feedforward=2 * self.hparams["model_dim"],
            num_heads=self.hparams["num_heads"],
            dropout=self.hparams["dropout"],
        )
        #
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams["model_dim"], self.hparams["model_dim"]),
            nn.LayerNorm(self.hparams["model_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams["dropout"]),
            nn.Linear(self.hparams["model_dim"], self.hparams["num_classes"]),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x


class ReverseDataset(Dataset):
    def __init__(self, num_categories, seq_len, size):
        super(ReverseDataset, self).__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size
        # [N, SeqLen]
        self.data = torch.randint(
            self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels


dataset = partial(ReverseDataset, 10, 16)
# shuffle, drop_last
num_workers = 0
ldm = libs_ml.LDataModule(dataset(50000), dataset(
    1000), dataset(10000), 128, num_workers)

inp_data, labels = ldm.train_dataloader.dataset[0]
print("Input data:", inp_data)
print("Labels:    ", labels)


max_epochs = 10
hparams = {
    "optim_name": "Adam",
    "model_name": "-",
    "optim_params": {
        "lr": 5e-4,
    },
    "model_params": {
        "input_dim": ldm.train_dataloader.dataset.num_categories,
        "model_dim": 32,
        "num_heads": 1,
        "num_classes": ldm.train_dataloader.dataset.num_categories,
        "num_layers": 1,
        "dropout": 0,
        "input_dropout": 0,
    },
    "lrs_params": {
        "warmup": 50,
        "T_max": max_epochs * len(ldm.train_dataloader),
    },
    "trainer_params": {
        "max_epochs": max_epochs
    },
    "clip_grad_norm_params": {
        "max_norm": 5,
        "norm_type": 2.
    }
}
model = MyModule(hparams["model_params"])
optimizer = optim.Adam(model.parameters(), **hparams["optim_params"])
# [Experiment]
# [Seq to Seq]

lmodel = MyLModule(model, optimizer, hparams)
trainer = libs_ml.Trainer(
    lmodel, True, runs_dir=RUNS_DIR, **hparams["trainer_params"])
trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
trainer.test(ldm.val_dataloader)
trainer.test(ldm.test_dataloader)
