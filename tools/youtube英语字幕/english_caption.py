import re
import os

if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    in_fpath = os.path.join(cur_dir, "./in.txt")
    out_fpath = os.path.join(cur_dir, "./out.txt")
    with open(in_fpath, "r") as f:
        text = f.read()

    ans = re.findall(r"<text.+?>(.+?)</text>", text)
    ans = " ".join(ans)
    ans = ans.replace("&#39;", "'")
    with open(out_fpath, "w") as f:
        f.write(ans)
