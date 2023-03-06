from ...._types import *


class Optimizer:
    def __init__(self, params: List[Parameter], defaults: Dict[str, Any]) -> None:
        self.defaults = defaults
        self.state = DefaultDict[Parameter, Dict[str, Tensor]](dict)
        self.param_groups: List[Dict[str, Any]] = []  # str含: params: List[Parameter], lr: float, ...

        params = list(params)  # Dict[str, List[Parameter]]
        param_groups = [{"params": params}]
        #
        for param_group in param_groups:
            self.add_param_group(param_group)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += "Parameter Group {0}\n".format(i)
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += "    {0}: {1}\n".format(key, group[key])
        format_string += ")"
        return format_string

    def state_dict(self) -> Dict[str, Any]:
        param_mappings: Dict[int, int] = {}
        start_index = 0

        def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update({id(p): i for i, p in enumerate(group["params"], start_index)
                                   if id(p) not in param_mappings})
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed
        param_groups: List[Dict[str, Any]] = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state: Dict[int, Dict[str, Tensor]] = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                                                      for k, v in self.state.items()}
        return {
            "state": packed_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = deepcopy(state_dict)
        groups = self.param_groups
        saved_groups: List[Dict[str, Any]] = state_dict["param_groups"]

        id_map: Dict[int, Parameter] = {old_id: p for old_id, p in
                                        zip(chain.from_iterable((g["params"] for g in saved_groups)),
                                            chain.from_iterable((g["params"] for g in groups)))}

        def cast(param: Parameter, value: Union[Tensor, Dict[str, Tensor]], 
                 key: Optional[str] = None) -> Union[Tensor, Dict[str, Tensor]]:
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, Tensor):
                if key != "step":
                    if param.is_floating_point():
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            else:
                return value

        state = DefaultDict[Parameter, Dict[str, Tensor]](dict)
        saved_state: Dict[int, Tensor] = state_dict["state"]
        for k, v in saved_state.items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)

        def update_group(group: Dict[str, Any], new_group: Dict[str, Any]) -> Dict[str, Any]:
            new_group["params"] = group["params"]
            return new_group
        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.state = state
        self.param_groups = param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        for group in self.param_groups:
            p: Parameter
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                    continue
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

    def step(self, closure: Callable[[], Tensor]) -> None:
        raise NotImplementedError

    def add_param_group(self, param_group: Dict[str, List[Parameter]]) -> None:

        params = param_group["params"]
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        #
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))
        assert param_set.isdisjoint(set(params))
        #
        self.param_groups.append(param_group)


# if __name__ == "__main__":
#     # from libs import *
#     model = tvm.resnet18().cuda()
#     o = optim.SGD(model.parameters(), 0.01, momentum=0.9)
#     x = torch.randn((16, 3, 100, 100)).cuda()
#     y = model(x).mean()
#     y.backward()
#     o.step()
#     o.zero_grad()
#     state_dict = o.state_dict()
#     # 
#     o = optim.SGD(model.parameters(), 0.01, momentum=0.9)
#     o.load_state_dict(state_dict)
#     print(o.param_groups[0]["params"][0].device)
