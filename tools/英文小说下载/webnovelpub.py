from libs import *


URL = "https://boxnovel.com/novel/lord-of-the-mysteries-boxnovel/chapter-1/"
i = 0
cur_dir = os.path.dirname(__file__)
fpath = os.path.join(cur_dir, "out.txt")
with open(fpath, "w") as f:
    pass
#
while True:
    resp = requests.get(URL)
    html: Element2 = etree.HTML(resp.text, None)
    #
    elem: Element2 = html.xpath("//div[@class=\"text-left\"]")[0]
    texts = libs_utils.xpath_get_text(elem)
    texts = re.sub(r"\n{2,}", "\n", texts).strip()
    texts = texts + "\n" * 3
    with open(fpath, "a") as f:
        f.write(texts)
    #
    next_pages: List[Element2] = html.xpath("//*[@id=\"manga-reading-nav-foot\"]//div[@class=\"nav-next \"]/a")
    if len(next_pages) == 0:
        break
    # 
    next_page: Element2 = next_pages[0]
    URL: str = next_page.attrib["href"]
    print(f"\r>> {i}, len(texts)={len(texts)}, next_url={URL}", end="")
    i += 1
print()
