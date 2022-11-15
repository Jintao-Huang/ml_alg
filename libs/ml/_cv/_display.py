# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date: 

import cv2 as cv
import numpy as np
from numpy import ndarray
from typing import Tuple, List, Union, Optional
from matplotlib.colors import to_rgb


__all__ = ["Box", "Point", "Color", "draw_box",
           "draw_target_in_image", "voc_labels", "generate_color", "resize_max"]

Box = Union[Tuple[int, int, int, int], List[int], ndarray]  # ltrb
Point = Union[Tuple[int, int], List[int], ndarray]
# BGR. Color可以使用f'C{i}'的方式指定. 共10个[0..9]
Color = Union[Tuple[int, int, int], List[int], str]


def draw_box(image: ndarray, box: Box, color: Color, is_bgr=True) -> None:
    # image: uint8
    if image.dtype != np.uint8:
        raise ValueError(f"image.dtype:{image.dtype}")
    if isinstance(color, str):
        c_f = to_rgb(color)
        c_f = reversed(c_f) if is_bgr else c_f
        color = tuple(int(c * 255) for c in c_f)
    #
    cv.rectangle(image, (box[0], box[1]),
                 (box[2], box[3]), color, thickness=2)

# if __name__ == "__main__":
#     x = cv.imread("asset/1.png")
#     draw_box(x, (0, 10, 100, 200), (0, 0, 0))
#     cv.imshow("1", x)
#     cv.waitKey()


def _draw_text(image: ndarray, box: Box, text: str, rect_color: Color, is_bgr=True) -> None:
    # text: 只支持英文
    # rect_color代表text的底色有一个rect. 它的颜色
    if image.dtype != np.uint8:
        raise ValueError(f"image.dtype:{image.dtype}")
    if isinstance(rect_color, str):
        c_f = to_rgb(rect_color)
        c_f = reversed(c_f) if is_bgr else c_f
        rect_color = tuple(int(c * 255) for c in c_f)
    #
    _thickness = 2  # 表示draw_box中的thickness
    _font_h, _font_w = 16, 10
    box_lt = (box[0] - _thickness // 2, box[1] - _font_h)
    #
    text_lb = (box[0], box[1] - 4)
    box_rd = (box[0] + int(len(text) * _font_w), box[1])
    cv.rectangle(image, box_lt, box_rd, rect_color, -1)
    cv.putText(image, text, text_lb, fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)


# if __name__ == "__main__":
#     x = cv.imread("asset/1.png")
#     box = (100, 200, 300, 400)
#     draw_box(x, box, (0, 0, 0))
#     _draw_text(x, box, "transformer", (0, 255, 0))
#     cv.imshow("2", x)
#     cv.waitKey()

def draw_target_in_image(image: ndarray, boxes: ndarray, labels: ndarray,
                         scores: Optional[ndarray], labels_str: Optional[List[str]], colors: Optional[List[Color]] = None,
                         is_bgr=True) -> None:
    """
    image: shape[H, W, C]. uint8
    boxes: shape[n_boxes, 4]. ltrb. int32
    labels: shape[n_boxes]. int32
    scores: shape[n_boxes]. float64. 从大到小排序. 可以为None
    labels_str: len[n_labels]. 
    colors: len[n_labels]. 如果colors为None, 则使用f"C{i}", 但只支持10个类
    """

    n_boxes = boxes.shape[0]
    n_labels = int(np.max(labels) + 1)
    labels_str = labels_str if labels_str is not None else \
        [str(i) for i in range(n_labels)]
    colors = colors if colors is not None else [
        f"C{i}" for i in range(n_labels)]

    # 先画框, 再写字, 防止覆盖
    # 因为scores从大到小排序, 所以我们保证大的不容易被覆盖: 反向遍历
    boxes = boxes.astype(np.int32)
    for i in reversed(range(n_boxes)):
        box = boxes[i]
        color: Color = colors[i]
        draw_box(image, box, color, is_bgr)
    for i in reversed(range(n_boxes)):
        box = boxes[i]
        color = colors[i]
        label = labels_str[int(labels[i])]
        text = f"{label}"
        text += "" if scores is None else f" {scores[i]:.2f}"
        _draw_text(image, box, text, color, is_bgr)


voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def generate_color(n: int) -> List[Color]:
    g = np.random.default_rng(42)
    colors = g.integers(0, 256, size=(n * 3)).reshape(n, 3).tolist()
    return colors


# if __name__ == "__main__":
#     import numpy as np
#     import cv2 as cv
#     x = cv.imread("asset/1.png")
#     boxes = np.array([[10, 10, 100, 200], [50, 50, 200, 100.]])
#     labels = np.array([1, 2.])
#     scores = np.array([1., 0.65])
#     labels_str = ["aaa", "bbb", "ccc"]
#     colors = generate_color(3)
#     # scores, labels_str = None, None
#     draw_target_in_image(x, boxes, labels, scores,
#                          labels_str, colors, is_bgr=True)
#     cv.imshow("1", x)
#     cv.waitKey()


def resize_max(image: ndarray, max_size: Tuple[Optional[int], Optional[int]],
               interpolation=cv.INTER_LINEAR) -> ndarray:
    # 将图像缩小到max_size以下
    # 若无限制, 则可以设置为None.
    #
    INF = 1e15
    h, w = image.shape[:2]
    max_w, max_h = max_size
    if max_w is None:
        max_w = INF
    if max_h is None:
        max_h = INF
    dx, dy = max_w / w, max_h / h  # aw <= max_w
    dxy = min(dx, dy)
    dsize = int(dxy * w), int(dxy * h)
    return cv.resize(image, dsize, interpolation=interpolation)


# if __name__ == "__main__":
#     import numpy as np
#     import cv2 as cv
#     x = cv.imread("asset/1.png")
#     x = cv.resize(x, (1000, 2000))
#     cv.imshow("1", x)
#     cv.waitKey()
#     x = resize_max(x, (1500, 1500))
#     print(x.shape)
#     cv.imshow("1", x)
#     cv.waitKey()
