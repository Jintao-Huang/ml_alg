
from libs import *

def convert_double_to_single(fpath: str, out_fpath: Optional[str]) -> None:
    if out_fpath is None:
        out_fpath = fpath
    with open(fpath, "r") as f:
        text = f.read()
    res = re.sub(r"'(.+?)'", r'"\1"', text)
    with open(out_fpath, "w") as f:
        f.write(res)


if __name__ == "__main__":
    convert_double_to_single("test.txt", "test2.txt")