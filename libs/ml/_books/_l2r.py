from ..._types import *
# from libs import *

def tf_idf(d: List[Dict[str, int]], i: int, s: str) -> float:
    """p8
    Ref: https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Example_of_tf%E2%80%93idf
    return tf_idf >= 0
    """
    len_doc = sum(v for c, v in d[i].items())
    n = 0
    if s in d[i]:
        n = d[i][s]
    tf = n / len_doc
    # 
    N_doc = len(d)
    cnt = 0
    for i in range(N_doc):
        if s in d[i]:
            cnt += 1
    idf = math.log10(N_doc / cnt)  # >=0
    return tf * idf

    

if __name__ == "__main__":
    d = [{"this": 1, "is": 1, "a": 2, "sample": 1},
         {"this": 1, "is": 1, "another": 2, "example": 3}]
    print(tf_idf(d, 0, "this"))
    print(tf_idf(d, 1, "this"))
    print(tf_idf(d, 0, "example"))
    print(tf_idf(d, 1, "example"))
"""
0.0
0.0
0.0
0.12901285528456335
"""