import time
from typing import Optional

from timm.metrics import ScalarAvgMinMax


class Tracker:

    def __init__(self):
        self.data_time = ScalarAvgMinMax()  # time for data loader to produce batch of samples
        self.step_time = ScalarAvgMinMax()  # time for model step
        self.iter_time = ScalarAvgMinMax()  # full iteration time incl. data, step, and book-keeping
        self.epoch_time = ScalarAvgMinMax()

        self.iter_timestamp: Optional[float] = None
        self.prev_timestamp: Optional[float] = None
        self.epoch_timestamp: Optional[float] = None

    def _measure_iter(self, ref_timestamp=None):
        timestamp = time.perf_counter()
        self.prev_timestamp = timestamp

    def mark_iter(self):
        timestamp = time.perf_counter()
        if self.iter_timestamp is not None:
            iter_time = timestamp - self.iter_timestamp
            self.iter_time.update(iter_time)
        self.iter_timestamp = self.prev_timestamp = timestamp

    def mark_iter_data_end(self):
        assert self.prev_timestamp is not None
        timestamp = time.perf_counter()
        data_time = timestamp - self.prev_timestamp
        self.data_time.update(data_time)
        self.prev_timestamp = timestamp

    def mark_iter_step_end(self):
        assert self.prev_timestamp is not None
        timestamp = time.perf_counter()
        step_time = timestamp - self.prev_timestamp
        self.step_time.update(step_time)
        self.prev_timestamp = timestamp

    def mark_epoch(self):
        timestamp = time.perf_counter()
        if self.epoch_timestamp is not None:
            epoch_time = timestamp - self.epoch_timestamp
            self.epoch_time.update(epoch_time)
        self.epoch_timestamp = timestamp

