try:
    from .pre import *
except ImportError:
    from pre import *
from matplotlib.colors import to_rgba


def basic_of_pytorch() -> None:
    _tensor()
    _dynamic_computation_graph_and_backpropagation()
    _gpu_support()


def _tensor() -> None:
    # [The Basics of PyTorch]
    print("Using torch", torch.__version__)
    torch.manual_seed(42)
    # [Tensors]
    x = torch.Tensor(2, 3, 4)
    print(x)
    # [Initialization]
    x = torch.Tensor([[1, 2], [3, 4]])
    print(x)

    x = torch.rand(2, 3, 4)
    print(x)

    shape = x.shape
    print("Shape:", shape)

    size = x.size()
    print("Size:", size)

    dim1, dim2, dim3 = x.size()
    print("Size:", dim1, dim2, dim3)
    # [Tensor to Numpy, and Numpy to Tensor]
    np_arr = np.array([[1, 2], [3, 4]])
    tensor = torch.from_numpy(np_arr)

    print("Numpy array:", np_arr)
    print("PyTorch tensor:", tensor)

    tensor = torch.arange(4)
    np_arr = tensor.numpy()

    print("PyTorch tensor:", tensor)
    print("Numpy array:", np_arr)

    np_arr = tensor.cpu().numpy()
    # [Operations]
    x1 = torch.rand(2, 3)
    x2 = torch.rand(2, 3)
    y = x1 + x2

    print("X1", x1)
    print("X2", x2)
    print("Y", y)

    x1 = torch.rand(2, 3)
    x2 = torch.rand(2, 3)
    print("X1 (before)", x1)
    print("X2 (before)", x2)

    x2.add_(x1)
    print("X1 (after)", x1)
    print("X2 (after)", x2)

    x = torch.arange(6)
    print("X", x)

    x = x.view(2, 3)
    print("X", x)

    x = x.permute(1, 0)
    print("X", x)

    x = torch.arange(6)
    x = x.view(2, 3)
    print("X", x)

    W = torch.arange(9).view(3, 3)
    print("W", W)

    h = torch.matmul(x, W)
    print("h", h)
    # [Indexing]
    x = torch.arange(12).view(3, 4)
    print("X", x)

    print(x[:, 1])
    print(x[0])
    print(x[:2, -1])
    print(x[1:3, :])


def _dynamic_computation_graph_and_backpropagation() -> None:
    # [Dynamic Computation Graph and Backpropagation]
    x = torch.ones((3,))
    print(x.requires_grad)

    x.requires_grad_(True)
    print(x.requires_grad)

    x = torch.arange(3, dtype=torch.float32, requires_grad=True)
    print("X", x)

    a = x + 2
    b = a ** 2
    c = b + 3
    y = c.mean()
    print("Y", y)

    y.backward()
    print(x.grad)


def _gpu_support() -> None:
    # [GPU support]
    gpu_avail = torch.cuda.is_available()
    print(f"Is the GPU available? {gpu_avail}")

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    x = torch.zeros(2, 3)
    x = x.to(device)
    print("X", x)

    x = torch.randn(5000, 5000)

    # start_time = time.time()
    _ = libs_utils.test_time(lambda: torch.matmul(x, x), number=5)
    _ = libs_utils.test_time(lambda: torch.matmul(
        x, x), number=5, timer=libs_ml.time_synchronize)
    # end_time = time.time()
    # print(f"CPU time: {(end_time - start_time):6.5f}s")

    if torch.cuda.is_available():
        x = x.to(device)
        _2 = libs_utils.test_time(lambda: torch.matmul(
            x, x), number=5, timer=libs_ml.time_synchronize).cpu()
        print(torch.allclose(_, _2))

    libs_ml.seed_everything(42, gpu_dtm=False)


def learning_by_example():
    """Continuous XOR"""
    # [Learning by example: Continuous XOR]
    # [The model]
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    libs_ml.seed_everything(42, gpu_dtm=False)

    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, x):
            pass

    # [Simple classifier]

    class SimpleClassifier(nn.Module):
        def __init__(self, num_inputs, num_hidden, num_outputs):
            super(SimpleClassifier, self).__init__()
            self.linear1 = nn.Linear(num_inputs, num_hidden)
            self.act_fn = nn.Tanh()
            self.linear2 = nn.Linear(num_hidden, num_outputs)

        def forward(self, x):
            x = self.linear1(x)
            x = self.act_fn(x)
            x = self.linear2(x)
            return x

    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    print(model)

    for name, param in model.named_parameters():
        print(f"Parameter {name}, shape {param.shape}")

    # [The data]
    from torch.utils import data

    # [The dataset class]
    dataset = libs_ml.XORDataset(200)
    print("Size of dataset:", len(dataset))
    print("Data point 0:", dataset[0])

    _, ax = plt.subplots(figsize=(4, 4), dpi=400)
    libs_ml.visualize_samples(dataset.data, dataset.labels, ax)
    plt.show()
    # [The data loader class]
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    data_inputs, data_labels = next(iter(data_loader))
    print("Data inputs", data_inputs.shape, "\n", data_inputs)
    print("Data labels", data_labels.shape, "\n", data_labels)
    # [Optimization]
    # [Loss modules]
    # [^数值稳定性]
    x = torch.Tensor([100, 100.])
    x.requires_grad_(True)
    x2 = torch.sigmoid(x)
    y = torch.Tensor([0, 0.])
    loss = nn.BCELoss()
    z = loss(x2, y)
    z.backward()
    print(z, x.grad)
    #
    x = torch.Tensor([100, 100.])
    x.requires_grad_(True)
    y = torch.Tensor([0, 0.])
    loss = nn.BCEWithLogitsLoss()
    z = loss(x, y)
    z.backward()
    print(z, x.grad)
    #
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # [Training]
    train_dataset = libs_ml.XORDataset(1000)
    train_data_loader = data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    ######

    class MyLModule(libs_ml.LModule):
        def __init__(self, model: Module, optim: Optimizer, hparams: Optional[Dict[str, Any]] = None) -> None:
            super(MyLModule, self).__init__(
                model, optim, hparams)
            self.loss_fn = nn.BCEWithLogitsLoss()


        def training_step(self, batch: Any) -> Tensor:
            # 返回的Tensor(loss)用于优化
            x_batch, y_batch = batch
            pred = self.model(x_batch)[:, 0]
            loss = self.loss_fn(pred, y_batch.float())
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch: Any) -> Union[Tensor, float]:
            # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
            x_batch, y_batch = batch
            pred = self.model(x_batch)[:, 0]
            loss = self.loss_fn(pred, y_batch.float())
            self.log("val_loss", loss)
            return -loss

        def test_step(self, batch: Any) -> None:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            y = y >= 0
            acc = libs_ml.accuracy_score(y, y_batch)
            self.log("test_acc", acc)
            
    runs_dir = os.path.join(PL_RUNS_DIR, "_1")
    lmodel = MyLModule(model, optimizer, {
        "model": "MLP_2", "optim": {"name": "SGD", "lr": 0.1}})
    ####
    trainer = libs_ml.Trainer(lmodel, True, 100, runs_dir)
    libs_utils.test_time(lambda: trainer.fit(
        train_data_loader, train_data_loader), number=1, warm_up=0)

    # [Saving a model]
    # [^test state_dict]
    state_dict = model.state_dict()
    print(state_dict)
    model.register_buffer("a", torch.Tensor([1, 2, 3.]))
    print(list(model.buffers()))
    print(model.state_dict())
    del model.a
    #
    state_dict = model.state_dict()
    print(state_dict)
    save_path = os.path.join(trainer.ckpt_dir, "m.ckpt")
    lmodel.save_checkpoint(save_path)
    new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    lmodel.model = new_model
    lmodel.load_from_checkpoint(save_path)
    print("Original model\n", model.state_dict())
    print("\nLoaded model\n", new_model.state_dict())
    # [Evaluation]

    test_dataset = libs_ml.XORDataset(500)
    test_data_loader = data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, drop_last=False)

    print(trainer.test(test_data_loader))
    _, ax = plt.subplots(figsize=(4, 4), dpi=400)
    libs_ml.plot_classification_map(
        lmodel.model, device, (-0.5, 1.5, -0.5, 1.5), 2, ax)
    libs_ml.visualize_samples(dataset.data, dataset.labels, ax)
    plt.show()


if __name__ == "__main__":
    # basic_of_pytorch()
    learning_by_example()


# Is the GPU available? True
# Device cuda
# X tensor([[0., 0., 0.],
#         [0., 0., 0.]], device='cuda:0')
# time: 0.530436±0.037361 |max: 0.586098 |min: 0.487177
# time: 0.708941±0.294348 |max: 1.218112 |min: 0.484378
# time: 0.114076±0.027328 |max: 0.167951 |min: 0.091875
# False
# SimpleClassifier(
#   (linear1): Linear(in_features=2, out_features=4, bias=True)
#   (act_fn): Tanh()
#   (linear2): Linear(in_features=4, out_features=1, bias=True)
# )
# Parameter linear1.weight, shape torch.Size([4, 2])
# Parameter linear1.bias, shape torch.Size([4])
# Parameter linear2.weight, shape torch.Size([1, 4])
# Parameter linear2.bias, shape torch.Size([1])
# Size of dataset: 200
# Data point 0: (tensor([0.8675, 0.9484]), tensor(0))
# Data inputs torch.Size([8, 2])
#  tensor([[ 1.1953,  0.2049],
#         [-0.1459,  0.8506],
#         [-0.1253,  0.1119],
#         [ 0.0531, -0.1361],
#         [ 0.1345,  0.0127],
#         [-0.1449,  0.9395],
#         [ 1.0506,  0.9082],
#         [ 1.0080,  0.0745]])
# Data labels torch.Size([8])
#  tensor([1, 1, 0, 0, 0, 1, 0, 1])
# tensor(100., grad_fn=<BinaryCrossEntropyBackward0>) tensor([0., 0.])
# tensor(100., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>) tensor([0.5000, 0.5000])
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 70.62it/s]
# OrderedDict([('linear1.weight', tensor([[ 1.8628,  1.7251],
#         [-2.6255,  2.0512],
#         [-0.1905,  0.1127],
#         [-1.6815,  2.4204]], device='cuda:0')), ('linear1.bias', tensor([-0.2069, -0.9411,  0.5876,  0.6650], device='cuda:0')), ('linear2.weight', tensor([[ 2.4035,  2.9415,  0.1604, -2.8211]], device='cuda:0')), ('linear2.bias', tensor([0.5864], device='cuda:0'))])
# [tensor([1., 2., 3.])]
# OrderedDict([('a', tensor([1., 2., 3.])), ('linear1.weight', tensor([[ 1.8628,  1.7251],
#         [-2.6255,  2.0512],
#         [-0.1905,  0.1127],
#         [-1.6815,  2.4204]], device='cuda:0')), ('linear1.bias', tensor([-0.2069, -0.9411,  0.5876,  0.6650], device='cuda:0')), ('linear2.weight', tensor([[ 2.4035,  2.9415,  0.1604, -2.8211]], device='cuda:0')), ('linear2.bias', tensor([0.5864], device='cuda:0'))])
# OrderedDict([('linear1.weight', tensor([[ 1.8628,  1.7251],
#         [-2.6255,  2.0512],
#         [-0.1905,  0.1127],
#         [-1.6815,  2.4204]], device='cuda:0')), ('linear1.bias', tensor([-0.2069, -0.9411,  0.5876,  0.6650], device='cuda:0')), ('linear2.weight', tensor([[ 2.4035,  2.9415,  0.1604, -2.8211]], device='cuda:0')), ('linear2.bias', tensor([0.5864], device='cuda:0'))])
# Original model
#  OrderedDict([('linear1.weight', tensor([[ 1.8628,  1.7251],
#         [-2.6255,  2.0512],
#         [-0.1905,  0.1127],
#         [-1.6815,  2.4204]], device='cuda:0')), ('linear1.bias', tensor([-0.2069, -0.9411,  0.5876,  0.6650], device='cuda:0')), ('linear2.weight', tensor([[ 2.4035,  2.9415,  0.1604, -2.8211]], device='cuda:0')), ('linear2.bias', tensor([0.5864], device='cuda:0'))])

# Loaded model
#  OrderedDict([('linear1.weight', tensor([[ 1.8628,  1.7251],
#         [-2.6255,  2.0512],
#         [-0.1905,  0.1127],
#         [-1.6815,  2.4204]])), ('linear1.bias', tensor([-0.2069, -0.9411,  0.5876,  0.6650])), ('linear2.weight', tensor([[ 2.4035,  2.9415,  0.1604, -2.8211]])), ('linear2.bias', tensor([0.5864]))])
# Accuracy of the model: 100.00%
