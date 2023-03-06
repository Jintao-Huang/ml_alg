from ...._types import *

_T = TypeVar("_T")


class Module:
    training: bool
    _parameters: Dict[str, Optional["Parameter"]]
    _buffers: Dict[str, Optional["Tensor"]]
    _non_persistent_buffers_set: Set[str]
    _modules: Dict[str, Optional["Module"]]

    def __init__(self) -> None:
        super().__setattr__("training", True)
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_modules", OrderedDict())

    forward: Callable[..., Any]

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        self._buffers[name] = tensor
        if not persistent:
            self._non_persistent_buffers_set.add(name)
        else:
            self._non_persistent_buffers_set.discard(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: "Module" = self
        for item in atoms:
            mod = getattr(mod, item)
        return mod

    def get_parameter(self, target: str) -> Parameter:
        module_path, _, param_name = target.rpartition(".")
        mod: "Module" = self.get_submodule(module_path)
        param: Parameter = getattr(mod, param_name)
        return param

    def get_buffer(self, target: str) -> Tensor:
        module_path, _, buffer_name = target.rpartition(".")
        mod: "Module" = self.get_submodule(module_path)
        assert buffer_name in self._buffers
        buffer: Tensor = getattr(mod, buffer_name)
        return buffer

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> "Module":
        for module in self.children():
            module._apply(fn)
        for key, param in self._parameters.items():
            if param is None:
                continue
            with torch.no_grad():
                param.data = fn(param)
            if param.grad is not None:
                with torch.no_grad():
                    param.grad.data = fn(param.grad)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def apply(self, fn: Callable[["Module"], None]) -> "Module":
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self, device: Optional[Union[int, Device]] = None) -> "Module":
        return self._apply(lambda t: t.cuda(device))

    def cpu(self) -> "Module":
        return self._apply(lambda t: t.cpu())

    def type(self, dst_type: Union[Dtype, str]) -> "Module":
        return self._apply(lambda t: t.type(dst_type))

    def float(self) -> "Module":
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self) -> "Module":
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self) -> "Module":
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def to(self, device: Optional[Union[int, Device]] = None, dtype: Optional[Union[Dtype, str]] = None,
           non_blocking: bool = False) -> "Module":
        def convert(t: Tensor) -> Tensor:
            return t.to(device, dtype, non_blocking)
        return self._apply(convert)

    def _call_impl(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def __getattr__(self, name: str) -> Union[Tensor, "Module", None]:
        if name in self._parameters:
            return self._parameters[name]
        if name in self. _buffers:
            return self._buffers[name]
        if name in self._modules:
            return self._modules[name]
        assert True  # 其他属性会在 __getattribute__中找到

    def __setattr__(self, name: str, value: Union[Parameter, Tensor, "Module", None]) -> None:
        def remove_from(name: str, *dicts_or_sets: Union[Dict[str, Any], Set[str]]):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        d.pop(name)
                    else:
                        d.remove(name)
        #
        params: Dict[str, Optional[Parameter]] = self._parameters
        if isinstance(value, Parameter):
            remove_from(name, self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
            return
        if name in params:
            assert value is None
            self.register_parameter(name, value)
            return
        #
        modules: Dict[str, Optional["Module"]] = self._modules
        if isinstance(value, Module):
            remove_from(name, self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
            modules[name] = value
            return
        if name in modules:
            assert value is None
            modules[name] = value
            return
        #
        buffers: Dict[str, Optional[Tensor]] = self._buffers
        if name in buffers:
            assert value is None or isinstance(value, torch.Tensor)
            buffers[name] = value
            return
        #
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            self._parameters.pop(name)
            return
        if name in self._buffers:
            self._buffers.pop(name)
            self._non_persistent_buffers_set.discard(name)
            return
        if name in self._modules:
            self._modules.pop(name)
            return
        super().__delattr__(name)

    def _save_to_state_dict(self, destination: OrderedDict[str, Tensor], prefix: str, keep_vars: bool):
        for name, param in self._parameters.items():
            if param is None:
                continue
            if not keep_vars:
                param = param.detach()
            destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is None or name in self._non_persistent_buffers_set:
                continue
            if not keep_vars:
                buf = buf.detach()
            destination[prefix + name] = buf

    def state_dict(self, *, destination: Optional[OrderedDict[str, Tensor]] = None,
                   prefix: str = "", keep_vars: bool = False) -> OrderedDict[str, Tensor]:

        if destination is None:
            destination = OrderedDict()

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is None:
                continue
            module.state_dict(destination=destination, prefix=f"{prefix}{name}.", keep_vars=keep_vars)
        return destination

    def _load_from_state_dict(self, state_dict:Dict[str, Tensor], prefix:str, strict:bool,
                              missing_keys:List[str], unexpected_keys:List[str]) -> None:
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                if strict:
                    missing_keys.append(key)
                continue
            # 
            input_param = state_dict[key]
            with torch.no_grad():
                param.copy_(input_param)

        if strict:
            for key in state_dict.keys():
                if not key.startswith(prefix):
                    continue
                input_name = key[len(prefix):]
                input_name = input_name.split(".", 1)[0]
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True) -> IncompatibleKeys:
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        state_dict = OrderedDict(state_dict)

        def load(module: "Module", local_state_dict: Dict[str, Tensor], prefix: str = "") -> None:
            module._load_from_state_dict(
                local_state_dict, prefix, True, missing_keys, unexpected_keys)
            for name, child in module._modules.items():
                if child is None:
                    continue
                child_prefix = f"{prefix}{name}."
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

        if strict:
            assert len(unexpected_keys) == 0 and len(missing_keys) == 0
        return IncompatibleKeys(missing_keys, unexpected_keys)

    def _named_members(self, get_members_fn: Callable[["Module"], dict_items[str, _T]],
                       prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, _T]]:
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                if module_prefix == "":
                    name = k
                else:
                    name = f"{module_prefix}.{k}"
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator["Module"]:
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        memo = set()
        for name, module in self._modules.items():
            if module is None and module in memo:
                continue
            memo.add(module)
            yield name, module

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set["Module"]] = None, prefix: str = "",
                      remove_duplicate: bool = True) -> Iterator[Tuple[str, "Module"]]:
        if memo is None:
            memo = set()
        if self in memo:
            return
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            if prefix == "":
                submodule_prefix = name
            else:
                submodule_prefix = f"{prefix}.{name}"
            for n, m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                yield n, m

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> "Module":
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.parameters():
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

    def _get_name(self) -> str:
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        #
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)
            child_lines.append(f"({key}): {mod_str}")
        lines = extra_lines + child_lines

        main_str = f"{self._get_name()}("
        if len(lines) > 0:
            if len(extra_lines) == 1 and len(child_lines) == 0:
                main_str += extra_lines[0]
            else:
                s = "\n  ".join(lines)
                main_str += f"\n  {s}\n"
        main_str += ")"
        return main_str


def _add_indent(s_: str, numSpaces: int) -> str:
    s = s_.split("\n")
    #
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    return f"{first}\n{s}"
