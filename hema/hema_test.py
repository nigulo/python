import numpy as np
import unittest
from hema import optimize
from data import Data
from conf import Conf

class test_hema(unittest.TestCase):
    
    def test_buy_low_sell_high(self):

        data = Data(sol=np.array([2.0, 1.0]), 
                    grid_buy=np.array([1.0, 2.0]), 
                    grid_sell=np.array([-0.9, -1.9]), 
                    fixed_cons=np.array([0.5, 0.4]), 
                    battery_start=0.5)
        conf = Conf(battery_min=0.1, 
                    battery_max=20)

        res = optimize(data, conf)
        print(res)

        battery_expected = np.array([19.5, -19.9])
        buy_expected = np.array([18, 0])
        sell_expected = np.array([0, 20.5])
        
        np.testing.assert_array_almost_equal(res.battery, battery_expected)
        np.testing.assert_array_almost_equal(res.buy, buy_expected)
        np.testing.assert_array_almost_equal(res.sell, sell_expected)

if __name__ == '__main__':
    unittest.main()
