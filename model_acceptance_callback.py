from typing import Any, List, Dict

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.logger import logger


class NERAcceptanceCallBack(keras.callbacks.Callback):

    def __init__(self,
                 model_save_dir: str,
                 kash_model: ABCTaskModel,
                 validate_data_x,
                 validate_data_y,
                 truncating: bool = False,
                 batch_size: int = 16,
                 monitor='f1-score',
                 mode='auto',
                 threshold=None,
                 history_record_mode='new') -> None:
        """
        Evaluate callback, calculate precision, recall and f1
        Args:
            kash_model: the kashgari task model to evaluate
            monitor: what the model save according to.
            mode: 'min' or 'max'
            history_record_mode: 'new' or 'keep'
        """
        super(NERAcceptanceCallBack, self).__init__()
        self.save_dir = model_save_dir
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        history_dir = f'{self.save_dir}/history'
        if not os.path.exists(history_dir):
            os.mkdir(history_dir)
        if history_record_mode == 'new':
            self.history_path = f'{history_dir}/history_{len(os.listdir(history_dir))}.csv'
        else:
            self.history_path = f'{history_dir}/history_{len(os.listdir(history_dir)) - 1}.csv'

        if not os.path.exists(self.history_path):
            self.perf = pd.DataFrame()
        else:
            self.perf = pd.read_csv(self.history_path)

        self.kash_model: ABCTaskModel = kash_model
        self.validate_data = [validate_data_x, validate_data_y]
        self.truncating = truncating
        self.batch_size = batch_size
        self.monitor = monitor
        self.mode = mode
        self.logs: List[Dict] = []

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = threshold or np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = threshold or -np.Inf
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
                self.best = threshold or np.Inf
            else:
                self.monitor_op = np.greater
                self.best = threshold or -np.Inf

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        print('\r\n')
        # evaluate validation data
        report = self.kash_model.evaluate(self.validate_data[0],  # type: ignore
                                          self.validate_data[1],
                                          truncating=self.truncating,
                                          batch_size=self.batch_size)

        self.logs.append({
            'precision': report['precision'],
            'recall': report['recall'],
            'f1-score': report['f1-score']
        })

        tf.summary.scalar('eval f1-score', data=report['f1-score'], step=epoch)
        tf.summary.scalar('eval recall', data=report['recall'], step=epoch)
        tf.summary.scalar('eval precision', data=report['precision'], step=epoch)
        print(f"epoch: {epoch} precision: {report['precision']:.6f},"
              f" recall: {report['recall']:.6f}, f1-score: {report['f1-score']:.6f}")

        # save history
        epoch_info = {'loss': [logs.get('loss')], 'accuracy': [logs.get('accuracy')],
                      'val_loss': [logs.get('val_loss')], 'val_accuracy': [logs.get('val_accuracy')],
                      'precision': [report['precision']], 'recall': [report['recall']], 'f1-score': [report['f1-score']]}
        if len(self.perf.columns) == 0:
            self.perf = pd.concat([self.perf, pd.DataFrame(epoch_info)], ignore_index=True)
        else:
            self.perf.loc[epoch] = {key: epoch_info[key][0] for key in epoch_info}
        self.perf.to_csv(self.history_path, index=False)
        logger.info(f'Successfully recorded this epoch into {self.history_path}')

        # save model
        current = epoch_info[self.monitor][0]
        try:
            if current is not None:
                if self.monitor_op(current, self.best):
                    logger.info('Epoch %d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s' % (epoch + 1, self.monitor,
                                                         self.best, current, self.save_dir))
                    self.best = current
                    self.kash_model.save(self.save_dir)
                    with open(os.path.join(self.save_dir, 'model_info.json'), mode='w+') as f:
                        json.dump(obj={'epoch': epoch,
                                       'monitor': self.monitor,
                                       'best': self.best},
                                  fp=f,
                                  indent=4)
                else:
                    logger.info('Epoch %d: %s did not improve from %0.5f' %
                                (epoch + 1, self.monitor, self.best))
            else:
                logger.warning('Monitor should be "loss", "accuracy", "val_loss", "val_accuracy",'
                               ' "precision", "recall", "f1-score". Saving failed.')
        except IOError as e:
            # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
            logger.exception('Errors happened when saving model.')
