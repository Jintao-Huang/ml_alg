# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ..._types import *
# from libs import *

__all__ = ["hf_get_state_dict"]
logger = ml.logger


def hf_get_state_dict(hf_home: str, model_id: str, commit_hash: str) -> Dict[str, Tensor]:
    model_id = model_id.replace("/", "--")
    model_fpath = os.path.join(hf_home, "hub", f"models--{model_id}", "snapshots", commit_hash, "pytorch_model.bin")
    state_dict = torch.load(model_fpath)
    return state_dict


def transformers_forward(ModelType: type, model_id: str, device: Optional[Device] = None, 
                         TokenizerType: Optional[type]=None, ConfigType: Optional[type]=None) -> None:
    if ConfigType is None:
        if hasattr(ModelType, "config_class"):
            ConfigType = ModelType.config_class
        else:
            ConfigType = AutoConfig
    if TokenizerType is None:
        TokenizerType = AutoTokenizer
    # 
    config = ConfigType.from_pretrained(model_id)
    model = ModelType.from_pretrained(model_id, config=config)
    model.eval()
    tokenizer: PreTrainedTokenizerBase = TokenizerType.from_pretrained(model_id)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    #
    x_str = "hello world"
    x = tokenizer(x_str, return_tensors="pt")
    if device is not None:
        x = ml.LModule.batch_to_device(x, device)
        model.to(device)
    with torch.no_grad():
        y = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
    logger.info(y.keys())
    return y


if __name__ == "__main__":
    device = ml.select_device([0])
    y = transformers_forward(AutoModelForMaskedLM, "roberta-base", device)
    logger.info(y.logits.shape)


def load_config_model_tokenizer(ModelType: type, model_id: str) -> Tuple[PretrainedConfig, Module, PreTrainedTokenizerBase]:
    config = AutoConfig.from_pretrained(model_id)
    model = ModelType.from_pretrained(model_id, config=config)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    #   
    return config, model, tokenizer


def GPT_out(model: Module, tokenizer: PreTrainedTokenizerBase,
              prompt: str, max_length: int, end_token_id_set: Optional[Set[int]] = None, 
              device: Optional[Device] = None) -> str:
    model.eval()
    if end_token_id_set is None:
        end_token_id_set = {tokenizer.eos_token_id, 198, 628}
    #
    x = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if device is not None:
        model.to(device)
        x = x.to(device)
    # 
    past_key_values = None
    res = []
    for _ in tqdm(range(max_length)):
        with torch.no_grad():
            y = model(x, past_key_values)
        past_key_values = y.past_key_values
        x = torch.argmax(y.logits[:, -1, :], dim=-1)[None]
        x_item = x.item()
        if x_item in end_token_id_set:
            if len(res) > 0:
                break
            else:
                continue
        res.append(x_item)
        # print(tokenizer.decode([x_item]), end="")
    return tokenizer.decode(res)

if __name__ == "__main__":
    ml.select_device([0])
    _, model, tokenizer = load_config_model_tokenizer(AutoModelForCausalLM, "gpt2")
    print(GPT_out(model, tokenizer, "hello! ", 50, device=Device(0)))