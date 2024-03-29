import numpy as np
import unittest
from hema import optimize
from data import Data
from conf import Conf
from result import Result

class test_hema(unittest.TestCase):

    def test_buy_eq_fixed_cons(self):

        data = Data(sol=np.array([0, 0, 0]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([2.0, 1.0, 0.5]), 
                    fixed_cons=np.array([2.5, 1.4, 3.2]), 
                    battery_start=0)
        conf = Conf(battery_max=20,
                    cons_max=0)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([2.5, 1.4, 3.2]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_fixed_cons_plus_sell(self):

        data = Data(sol=np.array([4, 1, 3]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([2.0, 1.0, 0.5]), 
                    fixed_cons=np.array([2.5, 1.4, 3.2]), 
                    battery_start=0)
        conf = Conf(battery_max=20,
                    cons_max=2)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([0, 0.4, 0.2]),
                              sell=np.array([1.5, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)
    
    def test_buy_low_sell_high(self):

        data = Data(sol=np.array([2.0, 1.0]), 
                    grid_buy=np.array([1.0, 2.0]), 
                    grid_sell=np.array([0.9, 1.9]), 
                    fixed_cons=np.array([0.5, 0.4]), 
                    battery_start=0.5)
        conf = Conf(battery_min=0.1, 
                    battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([19.5, -19.9]),
                              buy=np.array([18, 0]),
                              sell=np.array([0, 20.5]),
                              cons=np.array([0, 0]))        
        assert_result(res, expected_res)

    def test_neg_grid_buy(self):

        data = Data(sol=np.array([2.0, 1.0]), 
                    grid_buy=np.array([1.0, -2.0]), 
                    grid_sell=np.array([1.0, -2.0]), 
                    fixed_cons=np.array([2.5, 1.4]), 
                    battery_start=0.5)
        conf = Conf(battery_max=2,
                    buy_max=10,
                    cons_max=8)

        res = optimize(data, conf)
        print(res)

        # Here it would be more preferable solution where
        # battery = [-0.5, 2]
        # cons = [0, 7.6]
        expected_res = Result(battery=np.array([-0.5, 1.6]),
                              buy=np.array([0, 10]),
                              sell=np.array([0, 0]),
                              cons=np.array([0, 8]))        
        assert_result(res, expected_res)


def assert_result(res, expected_res):
        np.testing.assert_array_almost_equal(res.battery, expected_res.battery)
        np.testing.assert_array_almost_equal(res.buy, expected_res.buy)
        np.testing.assert_array_almost_equal(res.sell, expected_res.sell)
        np.testing.assert_array_almost_equal(res.cons, expected_res.cons)
    

if __name__ == '__main__':
    unittest.main()
