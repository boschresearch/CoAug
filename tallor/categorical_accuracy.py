#
#  This source code is from TALLOR
#    (https://github.com/JiachengLi1995/TALLOR)
#  This source code is licensed under the AGPL-3.0 license,
#  found in  the 3rd-party-licenses.txt file in the root directory of this source tree.
#

import torch

class CategoricalAccuracy:
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            print(
                "Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)"
            )
            assert 0
        if top_k <= 0:
            print("top_k passed to Categorical Accuracy must be > 0")
            assert 0
            
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            
                print("gold_labels must have dimension == predictions.size() - 1 but found tensor of shape: {}".format(predictions.size()))
                assert 0

        if (gold_labels >= num_classes).any():
            print(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )
            assert 0

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1)
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0

    def detach_tensors(self, *tensors):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

class MyCategoricalAccuracy(CategoricalAccuracy):
    ''' Wrapper of AllenNLP's CategoricalAccuracy with an additional used field '''
    def __init__(self, *args, **kwargs) -> None:
        super(MyCategoricalAccuracy, self).__init__(*args, **kwargs)
        self._used = False


    def __call__(self, *args, **kwargs):
        self._used = True
        super(MyCategoricalAccuracy, self).__call__(*args, **kwargs)


    def get_metric(self, reset: bool = False):
        if not self._used:
            return None
        return super(MyCategoricalAccuracy, self).get_metric(reset)

    def reset(self):
        super(MyCategoricalAccuracy, self).reset()
        self._used = False
