from libs import *

if __name__ == "__main__":
    from torchmetrics.functional import spearman_corrcoef as _spearman_corrcoef
    #
    target = torch.randn(1000 * 1000)
    preds = torch.randn(1000*1000)
    y = libs_ml.test_time(lambda: _spearman_corrcoef(preds, target))
    y2 = libs_ml.test_time(lambda: libs_ml.spearman_corrcoef(preds, target))
    print(torch.allclose(y, y2))
