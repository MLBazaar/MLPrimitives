from datetime import datetime
from unittest import TestCase

import pandas as pd

# from mlprimitives.adapters.pandas import resample


class ResamplpeTest(TestCase):

    def setup(self):
        self.df = pd.DataFrame({
            'dt': [
                datetime(2000, 1, day + 1, hour, 0)
                for day in range(31)
                for hour in range(24)
            ],
            'value': list(range(31 * 24))
        })

    def test_resample(self):
        pass
