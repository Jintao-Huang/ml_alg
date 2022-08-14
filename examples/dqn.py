# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:


from pre import *

logger = logging.getLogger(__name__)

"""
探索和训练是解耦的. 
1. 每一个iter. 会进行一次训练. 训练使用的数据集是从记忆库(memory pool)中随机采样的.
2. 同时Agent也会进行一次探索. 使用某些策略决策并采取行动(随机或模型决策). 从state变为next_state. 随后存入记忆库. 
  刚开始时会进行预热. 填部分的记忆入记忆库. 
"""


RENDER = True
RUNS_DIR = os.path.join(RUNS_DIR, "dqn")
DATASETS_PATH = os.environ.get(
    "DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
device_ids = [0]


class DQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


Memory = namedtuple(
    "Memory",
    # ndarray, RGB; int, float, bool, ndarray
    ["state", "action", "reward", "done", "next_state"]
)


class MemoryPool:
    """一个记忆库. """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.pool: Deque[Memory] = deque(maxlen=capacity)

    def len(self) -> int:
        return len(self.pool)

    def __getitem__(self, key) -> Memory:
        return self.pool[key]

    def add(self, memo: Memory) -> None:
        self.pool.append(memo)

    def sample(self) -> Memory:
        idx = np.random.choice(len(self.pool), (), replace=False)
        return self[idx]


class MyDataset(IterableDataset):
    def __init__(self, memo_pool: MemoryPool, dataset_len: int):
        self.memo_pool = memo_pool
        self.dataset_len = dataset_len

    def __iter__(self):
        # 返回生成器
        for _ in range(self.dataset_len):
            yield self.memo_pool.sample()

    def __len__(self):
        return self.dataset_len


class Agent:
    def __init__(self, env: Env, memo_pool: MemoryPool, model: Module, device: Union[str, Device]) -> None:
        self.env = env
        self.memo_pool = memo_pool
        self.model = model
        self.device = Device(device) if isinstance(device, str) else device
        #
        self.state = None
        self.reset_env()
        if RENDER:
            self.env.render()

    def reset_env(self) -> None:
        self.state = self.env.reset()

    def step(self, rand_p: float) -> Tuple[float, bool]:
        # 不返回state
        action = self._get_action(rand_p)
        next_state, reward, done, _ = self.env.step(action)
        memo = Memory(self.state, action, reward, done, next_state)
        self.memo_pool.add(memo)
        if done:
            self.reset_env()
        else:
            self.state = next_state
        if RENDER:
            self.env.render()
        return reward, done

    def _get_action(self, rand_p: float):
        if np.random.random() < rand_p:
            return self.env.action_space.sample()
        #
        state = torch.from_numpy(self.state)[None].to(self.device)
        q_value: Tensor = self.model(state)[0]
        return q_value.argmax(dim=0).item()


def get_rand_p(global_step, T_max: int, eta_min: float, eta_max: float):
    rand_p = libs_ml.cosine_annealing_lr(
        global_step, T_max, eta_min, [eta_max])[0]
    return rand_p


class MyLModule(libs_ml.LModule):
    def __init__(self, model: Module, optim: Optimizer, loss_fn: Module, agent: Agent, get_rand_p: Callable[[int], float],
                 hparams: Optional[Dict[str, Any]] = None) -> None:
        super(MyLModule, self).__init__(model, optim, hparams)
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        # self.optim, self.model
        # 模型训练会使用new_model和old_model. 在计算next_state的reward预测使用old_model.
        #   探索和训练使用new_model. 原因是: 消除关联性. 每sync_steps步, 进行同步
        self.old_model = deepcopy(self.model)
        self.loss_fn = loss_fn
        self.agent = agent
        self.get_rand_p = get_rand_p
        #
        self.warmup_memory_steps = self.hparams["warmup_memory_steps"]
        self.sync_steps = self.hparams["sync_steps"]
        self.gamma = self.hparams["gamma"]  # reward的衰减系数

        #
        self._warmup_memo(self.warmup_memory_steps)
        self.episode_reward = 0  # 一局的reward

    def training_epoch_start(self) -> None:
        super(MyLModule, self).training_epoch_start()
        self.old_model.train()
        self.old_model.to(self.device)

    def _warmup_memo(self, steps: int):
        for _ in tqdm(range(steps), desc=f"Warmup: "):
            self.agent.step(rand_p=1)

    def _train_step(self, batch: Any) -> Tensor:
        states, actions, rewards, dones, next_states = batch
        q_values = self.model(states)
        # 将states采取actions的q值与
        #   reward+gamma*next_states(dones则为0)采取最好行为进行逼近
        y_pred = q_values[torch.arange(len(actions)), actions]
        with torch.no_grad():
            y_true: Tensor = self.old_model(next_states).max(1)[0]
            y_true[dones] = 0.
        y_true.mul_(self.gamma).add_(rewards)
        loss = self.loss_fn(y_pred, y_true)
        return loss

    @torch.no_grad()
    def _agent_step(self) -> Tuple[float, bool]:
        rand_p = self.get_rand_p(self.global_step)  # 从1开始
        reward, done = self.agent.step(rand_p)
        if done:
            self.episode_reward = 0
        else:
            self.episode_reward += reward
        return reward, done

    def training_step(self, batch: Any) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化
        if self.global_step % self.sync_steps == 0:
            # load_state_dict是copy
            self.old_model.load_state_dict(self.model.state_dict())

        # train
        loss = self._train_step(batch)
        # step
        reward, done = self._agent_step()
        # log
        self.log("reward", reward, prog_bar_mean=False)
        self.log("done", done, prog_bar_mean=False)
        self.log("episode_reward", self.episode_reward)
        self.log("loss", loss)
        return loss


if __name__ == "__main__":
    libs_ml.seed_everything(42, gpu_dtm=False)
    hparams = {
        "memo_capacity": 1000,
        "dataset_len": 5000,
        "env_name": "CartPole-v1",
        "model_hidden_size": 128,
        "optim_name": "AdamW",
        "dataloader_hparams": {"batch_size": 32},
        "optim_hparams": {"lr": 1e-2, "weight_decay": 1e-5},  #
        "trainer_hparams": {"max_epochs": 10, "gradient_clip_norm": 20},
        #
        "rand_p": {
            "eta_max": 1,
            "eta_min": 0,
            "T_max": ...
        },
        "sync_steps": 20,  # old_model的同步的频率
        "warmup_memory_steps": 1000,  # 预热memory
        "gamma": 0.99,  # reward衰减

    }
    memo_pool = MemoryPool(hparams["memo_capacity"])
    dataset = MyDataset(memo_pool, hparams["dataset_len"])
    ldm = libs_ml.LDataModule(
        dataset, None, None, **hparams["dataloader_hparams"], shuffle_train=False, num_workers=0)
    hparams["rand_p"]["T_max"] = len(
        ldm.train_dataloader) * hparams["trainer_hparams"]["max_epochs"]

    env = gym.make(hparams["env_name"])
    in_channels: int = env.observation_space.shape[0]
    out_channels: int = env.action_space.n
    model = DQN(in_channels, out_channels, hparams["model_hidden_size"])
    agent = Agent(env, memo_pool, model, libs_ml.select_device(device_ids))

    #
    get_rand_p = partial(get_rand_p, **hparams["rand_p"])
    optimizer = getattr(optim, hparams["optim_name"])(
        model.parameters(), **hparams["optim_hparams"])
    runs_dir = CHECKPOINTS_PATH
    loss_fn = nn.MSELoss()

    lmodel = MyLModule(model, optimizer, loss_fn, agent, get_rand_p, hparams)
    trainer = libs_ml.Trainer(
        lmodel, device_ids, runs_dir=runs_dir, **hparams["trainer_hparams"])
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
