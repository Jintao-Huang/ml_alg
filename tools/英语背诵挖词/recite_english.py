# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from libs import *


prob = 0.5  # 空的概率
seed = None  # 随机种子
np.random.seed(seed)


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    in_fpath = os.path.join(cur_dir, "./in.txt")
    out_fpath = os.path.join(cur_dir, "./out.txt")
    with open(in_fpath, "r") as f:
        text = f.read()
    text = text.replace("\n", " ")
    words = [w for w in text.split(" ") if w != ""]
    idxs = np.random.permutation(len(words))[:int(len(words) * prob)]
    sorted_idxs = sorted(idxs)
    # 对标点符号问题进行改进
    ans = []  # 答案
    for i in sorted_idxs:
        w = words[i]
        if w[-1] in [".", ","]:
            ans.append(w[:-1])
            words[i] = "_" * (len(w) - 1) + w[-1]
        else:
            ans.append(w)
            words[i] = "_" * len(w)
    #
    res = " ".join(words)
    res += "\n\n\n\n\n\n\n\n"
    res += " ".join(ans)

    with open(out_fpath, "w") as f:
        f.write(res)










#




"""
hegemonic
inadequate
impoverished philosophy
vacuous theology
radical
individualism
institution
associations
institutional
interpreted
pattern
dignity
moral autonomy
context
afloat
tragedy
habitually
void
"""



"""
we believe that much of the thinking about the self of educated Americans
thinking that has become almost hegemonic in our universities and much of the
middle class, is based on inadequate social science, impoverished philosophy,
and vacuous theology. There are truths we do not see when we adopt the language of
radical individualism. We find ourselves not independently of other people and
institutions but through them. We never get the bottom of our selves on our own.
We discover who we are face to face and side by side with others in work, love,
and learning. All of our activity goes on in relationships, groups, associations,
and communities ordered by institutional structure and interpreted by cultural
patterns of meaning. Our individualism is itself one such pattern.

and the positive side of our individualism, our sense of the dignity, worth, and
moral autonomy of the individual, is dependent in a thousand ways in a social, cultural
and institutional context that keep us afloat even when we cannot very well describe it.
There is much in our life that we do not control, that we are not even responsible
for, that we receive as grace or face as tragedy, things Americans habitually prefer
not to think about,
Finally, we are not simply ends in ourselves, either as individuals or as a society
we are parts of the larger whole that we neither forget or imagine in our own image
without a high price.
If we are not to have a self that hangs in the void, slowly twisting in the wind
there are issues we cannot ignore

"""