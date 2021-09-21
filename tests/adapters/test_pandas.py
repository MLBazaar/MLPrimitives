from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from mlprimitives.adapters.pandas import resample


class ResampleTest(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'dt': [
                datetime(2000, 1, day + 1, hour, 0)
                for day in range(4)
                for hour in range(24)
            ],
            'value': np.arange(0, 4 * 24, dtype=float)
        })

    def test_resample_rule_str(self):

        out = resample(self.df.set_index('dt'), '1d')

        assert_frame_equal(out, pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 11.5},
            {'dt': datetime(2000, 1, 2), 'value': 35.5},
            {'dt': datetime(2000, 1, 3), 'value': 59.5},
            {'dt': datetime(2000, 1, 4), 'value': 83.5},
        ]))

    def test_resample_rule_int(self):

        out = resample(self.df.set_index('dt'), 86400)

        assert_frame_equal(out, pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 11.5},
            {'dt': datetime(2000, 1, 2), 'value': 35.5},
            {'dt': datetime(2000, 1, 3), 'value': 59.5},
            {'dt': datetime(2000, 1, 4), 'value': 83.5},
        ]))

    def test_resample_groupby(self):

        self.df['group1'] = ['A', 'B'] * 2 * 24
        self.df['group2'] = ['C', 'C', 'D', 'D'] * 24

        out = resample(self.df.set_index('dt'), '1d', groupby=['group1', 'group2'])

        assert_frame_equal(out, pd.DataFrame([
            {'group1': 'A', 'group2': 'C', 'dt': datetime(2000, 1, 1), 'value': 10.0},
            {'group1': 'A', 'group2': 'C', 'dt': datetime(2000, 1, 2), 'value': 34.0},
            {'group1': 'A', 'group2': 'C', 'dt': datetime(2000, 1, 3), 'value': 58.0},
            {'group1': 'A', 'group2': 'C', 'dt': datetime(2000, 1, 4), 'value': 82.0},
            {'group1': 'A', 'group2': 'D', 'dt': datetime(2000, 1, 1), 'value': 12.0},
            {'group1': 'A', 'group2': 'D', 'dt': datetime(2000, 1, 2), 'value': 36.0},
            {'group1': 'A', 'group2': 'D', 'dt': datetime(2000, 1, 3), 'value': 60.0},
            {'group1': 'A', 'group2': 'D', 'dt': datetime(2000, 1, 4), 'value': 84.0},
            {'group1': 'B', 'group2': 'C', 'dt': datetime(2000, 1, 1), 'value': 11.0},
            {'group1': 'B', 'group2': 'C', 'dt': datetime(2000, 1, 2), 'value': 35.0},
            {'group1': 'B', 'group2': 'C', 'dt': datetime(2000, 1, 3), 'value': 59.0},
            {'group1': 'B', 'group2': 'C', 'dt': datetime(2000, 1, 4), 'value': 83.0},
            {'group1': 'B', 'group2': 'D', 'dt': datetime(2000, 1, 1), 'value': 13.0},
            {'group1': 'B', 'group2': 'D', 'dt': datetime(2000, 1, 2), 'value': 37.0},
            {'group1': 'B', 'group2': 'D', 'dt': datetime(2000, 1, 3), 'value': 61.0},
            {'group1': 'B', 'group2': 'D', 'dt': datetime(2000, 1, 4), 'value': 85.0},
        ], columns=['group1', 'group2', 'dt', 'value']))

    def test_resample_on(self):

        out = resample(self.df, '1d', on='dt')

        assert_frame_equal(out, pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 11.5},
            {'dt': datetime(2000, 1, 2), 'value': 35.5},
            {'dt': datetime(2000, 1, 3), 'value': 59.5},
            {'dt': datetime(2000, 1, 4), 'value': 83.5},
        ]))

    def test_resample_reset_index_false(self):

        out = resample(self.df.set_index('dt'), '1d', reset_index=False)

        assert_frame_equal(out.reset_index(), pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 11.5},
            {'dt': datetime(2000, 1, 2), 'value': 35.5},
            {'dt': datetime(2000, 1, 3), 'value': 59.5},
            {'dt': datetime(2000, 1, 4), 'value': 83.5},
        ]))

    def test_resample_aggregation_str(self):

        out = resample(self.df.set_index('dt'), '1d', aggregation='max')

        assert_frame_equal(out, pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 23.0},
            {'dt': datetime(2000, 1, 2), 'value': 47.0},
            {'dt': datetime(2000, 1, 3), 'value': 71.0},
            {'dt': datetime(2000, 1, 4), 'value': 95.0},
        ]))

    def test_resample_aggregation_func(self):

        out = resample(self.df.set_index('dt'), '1d', aggregation=np.max)

        assert_frame_equal(out, pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 23.0},
            {'dt': datetime(2000, 1, 2), 'value': 47.0},
            {'dt': datetime(2000, 1, 3), 'value': 71.0},
            {'dt': datetime(2000, 1, 4), 'value': 95.0},
        ]))

    def test_resample_aggregation_import(self):

        out = resample(self.df.set_index('dt'), '1d', aggregation='numpy.max')

        assert_frame_equal(out, pd.DataFrame([
            {'dt': datetime(2000, 1, 1), 'value': 23.0},
            {'dt': datetime(2000, 1, 2), 'value': 47.0},
            {'dt': datetime(2000, 1, 3), 'value': 71.0},
            {'dt': datetime(2000, 1, 4), 'value': 95.0},
        ]))
