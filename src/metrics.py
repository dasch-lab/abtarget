import math
import torch

EPSILON = 1e-7

class MCC():
  def __init__(self, multidim_average='global'):
    self._multidim_average = multidim_average
    self._stats = {
      'tp': 0,
      'tn': 0,
      'fp': 0,
      'fn': 0
    }

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    # Element-wise division of the 2 tensors returns a new tensor which holds a

    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    

    '''
    Calculate the Matthew correlation coefficient for a pair of tensors.
    Calculated on the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
      - 1 and 1 (True Positive)
      - 1 and 0 (False Positive)
      - 0 and 0 (True Negative)
      - 0 and 1 (False Negative)
    '''
    confusion_vector = preds / target

    # sum_dim = [0, 1] if self._multidim_average == "global" else 1
    # self._stats['tp'] += ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    # self._stats['fn'] = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    # self._stats['fp'] = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    # self._stats['tn'] = ((target == preds) & (target == 0)).sum(sum_dim).squeeze().
    self._stats['tp'] += torch.sum(confusion_vector == 1).item()
    self._stats['fp'] += torch.sum(confusion_vector == float('inf')).item()
    self._stats['tn'] += torch.sum(torch.isnan(confusion_vector)).item()
    self._stats['fn'] += torch.sum(confusion_vector == 0).item()

    # Calculate MCC
    up = self._stats['tp']*self._stats['tn'] - self._stats['fp']*self._stats['fn']
    down = math.sqrt(
      (self._stats['tp']+self._stats['fp']) * 
      (self._stats['tp']+self._stats['fn']) * 
      (self._stats['tn']+self._stats['fp']) * 
      (self._stats['tn']+self._stats['fn'])
    )
      
    mcc = up / (down+EPSILON)
    #mcc = tf.where(tf.is_nan(mcc), tf.zeros_like(mcc), mcc)
    return mcc