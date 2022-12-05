
from libs import *
logger = libs_ml.logger

proxies = {
    'http': '127.0.0.1:7890',
    'https': '127.0.0.1:7890'
}
headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
}


def update_cite_num(fpath: str, ask: bool = True) -> int:
    """这个函数对谷歌学术的人机验证毫无办法. 哈哈哈...
    fname的格式:【{date}】{paper_name}[{c0}].pdf. 
      e.g.【2014_】GloVe[27682].pdf     
          【2103】Transformer in Transformer[295].pdf
    return: 返回0表示成功修改, 返回-1表示失败或未修改
    """
    # 我们需要获取fpath的文章信息, 原始引用信息
    # 随后我们查找第一个, 并获取其引用信息. 如果引用上升, 则更新.
    #  否则打印异常, 并不修改
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"fpath: {fpath}")
    dir_, fname = os.path.split(fpath)  # 返回目录和文件名
    m = re.match(r"【(.+?)】(.+?)\[(\d+?)\]", fname)
    if m is None:
        logger.info(f"异常: fname: {fname}, 请修改合适的文件名")
        return -1
    date, paper_name, n_cite = m.groups()
    params = {
        "q": paper_name,
        "hl": "zh-CN"
    }
    url = "https://scholar.google.com/scholar"
    try:
        req = requests.get(url, proxies=proxies,
                           params=params, headers=headers)
    except Exception as e:
        print(f"获取http异常: {e}")
        return -1
    #
    if req.status_code != 200:
        print(f"req.status_code: {req.status_code}, req.url: {req.url}")
        return -1
    text = req.text
    text = re.sub(r"</?[bi]>", "", text)  # 去掉 <b> <i>等

    #
    pn_list = re.findall(
        r'<a id=".+?" href=".+?" data-clk=".+?" data-clk-atid=".+?">(.+?)</a>', text)
    pn_list = [s for s in pn_list if "[PDF]" not in s]
    c_list = re.findall(r"被引用次数：(\d+)", text)
    assert len(pn_list) == len(c_list)
    pn0 = pn_list[0]
    c0 = c_list[0]
    del pn_list, c_list
    #
    if n_cite > c0:
        print(f"异常: fname: {fname}, pn0: {pn0}, c0: {c0}, 请修改合适的文件名")
        return -1
    #
    new_fname = f"【{date}】{paper_name}[{c0}].pdf"
    print(f'"{fname}" -> "{new_fname}"')
    if ask:
        yn = input(f"    论文名: {pn0}. 是否修改? (y/n)")
    else:
        yn = "y"
    if yn.lower() != "y":
        return -1
    #
    if fname == new_fname:
        print("    引用数未变, 无需修改")
    else:
        new_fpath = os.path.join(dir_, new_fname)
        print("    已修改")
        os.rename(fpath, new_fpath)
    return 0


if __name__ == "__main__":
    update_cite_num(
        "/home/jintao/Desktop/Transformer-xl/Hierarchy/【2103】Transformer in Transformer[295].pdf")
    update_cite_num(
        "/home/jintao/Desktop/2022-4-28[ET]/paper/PTM/【2014_】GloVe[29360].pdf")
