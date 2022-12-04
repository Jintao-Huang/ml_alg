from libs import *
FPATH = f"/home/jintao/Desktop/coding/python/ustcml/mini-lightning/runs"

def startswith_in(s: str, prefix_set: Set[str]) -> bool:
    for prefix in prefix_set:
        if s.startswith(prefix):
            return True
    return False


def replace_ckpt(
    dir_: str,
    #
    last: bool = True, best: bool = False
) -> None:
    """
    使用空文件(保持文件名), 取替代ckpt文件. 
        只处理checkpoints文件夹里的. 
    """
    prefix_set = set()
    if last:
        prefix_set.add("last")
    if best:
        prefix_set.add("best")
    iter = libs_utils.mywalk(dir_)
    for level, curr_dir, folder_list, fname_list in iter:
        dir_name = os.path.basename(curr_dir)
        if dir_name == "checkpoints":
            for fname in fname_list:
                if startswith_in(fname, prefix_set):
                    fpath = os.path.join(curr_dir, fname)
                    os.remove(fpath)
                    #
                    with open(fpath, "w"):
                        pass
                    #
                    libs_ml.logger.info(f"fpath: `{fpath}`")


if __name__ == "__main__":
    replace_ckpt(FPATH)
