from libs import *

if __name__ == "__main__":
    from torchmetrics.functional import spearman_corrcoef as _spearman_corrcoef
    #
    target = torch.randn(1000 * 1000).cuda()
    preds = torch.randn(1000*1000).cuda()
    y = libs_ml.test_time(lambda: _spearman_corrcoef(preds, target), 1, 1, libs_ml.time_synchronize)
    y2 = libs_ml.test_time(lambda: libs_ml.spearman_corrcoef(preds, target), 1, 1, libs_ml.time_synchronize)
    print(torch.allclose(y, y2))

"""
[INFO: mini-lightning] time[number=1]: 1.348975±0.000000, max=1.348975, min=1.348975
[INFO: mini-lightning] time[number=1]: 0.009190±0.000000, max=0.009190, min=0.009190
"""
