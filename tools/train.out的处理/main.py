import os
import re

if __name__ == "__main__":
    # 从文件中不断读取行, 如果以`Epoch \d:`, `   Val:`, `Test Last:`, `Test Best:`开头, 则记忆但不加入res.
    # 如果不一这个开头且是第一个, 则将记忆加入res. 并将改行加入res.
    cur_dir = os.path.dirname(__file__)
    in_fpath = os.path.join(cur_dir, "./train.out")
    out_fpath = os.path.join(cur_dir, "./train.out2")
    # 
    res = []
    need_save = False
    s = ""
    with open(in_fpath, "r") as f:
        for line in f:
            if re.match(r"Epoch \d:", line) or line.startswith("  Val:") or line.startswith("Test Last:") or line.startswith("Test Best:"):
                s = line
                need_save = True
            elif need_save:
                res.append(s)
                res.append(line)
                need_save = False
            else:
                res.append(line)
        if need_save:
            res.append(s)
            need_save = False
    text = "".join(res)
    with open(out_fpath, "w") as f:
        f.write(text)





