import tensorflow as tf


class AoaKerasProgressCallback(tf.keras.callbacks.Callback):

    def __init__(self, aoa_progress_callback):
        self.aoa_progress_callback = aoa_progress_callback
        self.iter = 0
        self.epoc = 0
        self.score = 0

    def on_epoch_end(self, epoc, logs={}):
        self.epoc = epoc
        self.score = logs.get("acc", 0)
        self._send_callback()

    def on_batch_end(self, batch, logs={}):
        self.iter = batch
        self.score = logs.get("acc", 0)
        self._send_callback()

    def _send_callback(self):
        self.aoa_progress_callback(iter=self.iter, epoc=self.epoc, score=self.score.item())
