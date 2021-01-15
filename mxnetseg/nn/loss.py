# coding=utf-8

import math
import gluoncv.loss as gloss
from mxnet.gluon.loss import Loss

__all__ = ['MixSoftmaxCrossEntropyLoss', 'FocalLoss']


class MixSoftmaxCrossEntropyLoss(gloss.SoftmaxCrossEntropyLoss):
    """
    SoftmaxCrossEntropyLoss with multiple aux weight.
    """

    def __init__(self, aux=False, aux_weight=None, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_label=ignore_label, **kwargs)
        self.aux = aux
        if aux and aux_weight:
            if isinstance(aux_weight, float):
                self.aux_weight = (aux_weight,)
            elif isinstance(aux_weight, (list, tuple)):
                self.aux_weight = tuple(aux_weight)
            else:
                raise RuntimeError(
                    f"Unknown type of aux_weight: {type(aux_weight)}, "
                    f"which should be float or list of float")

    def _aux_forward(self, F, *inputs, **kwargs):
        assert len(self.aux_weight) == (len(inputs) - 2)
        label = inputs[-1]
        total_loss = super(MixSoftmaxCrossEntropyLoss, self).hybrid_forward(F, inputs[0], label)
        pairs = self._get_pairs(inputs)
        for weight, pred in pairs:
            this_loss = super(MixSoftmaxCrossEntropyLoss, self).hybrid_forward(F, pred, label)
            total_loss = total_loss + weight * this_loss

        return total_loss

    def _get_pairs(self, inputs):
        pairs = []
        for i in range(len(self.aux_weight)):
            pair = (self.aux_weight[i], inputs[i + 1])
            pairs.append(pair)
        return tuple(pairs)

    def hybrid_forward(self, F, *inputs, **kwargs):
        """ input order: pred1, pred2, ..., label """
        if self.aux:
            return self._aux_forward(F, *inputs, **kwargs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).hybrid_forward(F, *inputs, **kwargs)


class FocalLoss(Loss):
    """
    Focal loss for semantic segmentation.
    Reference: Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.
    """

    def __init__(self, alpha=0.25, gamma=2, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label):
        pt = F.SoftmaxOutput(pred, label.astype(pred.dtype),
                             ignore_label=self._ignore_label,
                             multi_output=self._sparse_label,
                             use_ignore=True,
                             normalization='valid' if self._size_average else 'null')
        if self._sparse_label:
            loss = -F.pick(self._alpha * ((1 - pt) ** self._gamma) * F.log(pt),
                           label, axis=1, keepdims=True)
        else:
            raise NotImplementedError("Set sparse_label=True for semantic segmentation.")
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
