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
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([2.5, 1.4, 3.2]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_eq_sell_limited_buy_battery_max(self):

        data = Data(sol=np.array([0, 0, 0]), 
                    grid_buy=np.array([1.0, 1.1, 0.9]), 
                    grid_sell=np.array([1.0, 1.1, 0.9]), 
                    fixed_cons=np.array([0, 0, 0]), 
                    battery_start=10)
        conf = Conf(battery_max=10)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, -10, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 10, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_eq_sell(self):

        data = Data(sol=np.array([0, 0, 0]), 
                    grid_buy=np.array([1.1, 1.2, 0.9]), 
                    grid_sell=np.array([1.0, 1.1, 0.9]), 
                    fixed_cons=np.array([0, 0, 0]), 
                    battery_start=10)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, -10, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 10, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_plus_buy_eq_sell(self):

        data = Data(sol=np.array([0, 0, 0]), 
                    grid_buy=np.array([1.0, 1.1, 0.9]), 
                    grid_sell=np.array([1.0, 1.1, 0.9]), 
                    fixed_cons=np.array([0, 0, 0]), 
                    battery_start=10)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        # Here the first buy does not give any other benefit than just charging the battery
        expected_res = Result(battery=np.array([10, -20, 0]),
                              buy=np.array([10, 0, 0]),
                              sell=np.array([0, 20, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_eq_fixed_cons(self):

        data = Data(sol=np.array([0, 0, 0]), 
                    grid_buy=np.array([1.1, 2.2, 1.5]), 
                    grid_sell=np.array([0, 0, 0]), 
                    fixed_cons=np.array([2, 3, 5]), 
                    battery_start=10)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([-2, -3, -5]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_eq_fixed_cons_plus_sell(self):

        data = Data(sol=np.array([0, 0, 0]), 
                    grid_buy=np.array([1.2, 1.3, 1.2]), 
                    grid_sell=np.array([1.0, 1.1, 0.9]), 
                    fixed_cons=np.array([2, 2, 4]), 
                    battery_start=10)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([-2, -4, -4]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_fixed_cons(self):

        data = Data(sol=np.array([4, 1, 3]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([2.0, 1.0, 0.5]), 
                    fixed_cons=np.array([4, 1, 3]), 
                    battery_start=0)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_sell(self):

        data = Data(sol=np.array([4, 1, 3]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([2.0, 1.0, 0.5]), 
                    fixed_cons=np.array([0, 0, 0]), 
                    battery_start=0)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([4, 1, 3]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_fixed_cons_plus_sell(self):

        data = Data(sol=np.array([4, 1, 3.1]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([2.0, 1.0, 0.5]), 
                    fixed_cons=np.array([2.5, 1, 3]), 
                    battery_start=0)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([1.5, 0, 0.1]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_battery(self):

        data = Data(sol=np.array([4, 1, 3]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([0, 0, 0]), 
                    fixed_cons=np.array([0, 0, 0]), 
                    battery_start=0)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([4, 1, 3]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_battery_plus_sell(self):

        data = Data(sol=np.array([4, 1, 10]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([1.9, 0.9, 0.5]), 
                    fixed_cons=np.array([0, 0, 0]), 
                    battery_start=0)
        conf = Conf(battery_max=20,
                    sell_max=3.5)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0.5, -0.5, 6.5]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([3.5, 1.5, 3.5]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_battery_plus_fixed_cons(self):

        data = Data(sol=np.array([1, 2, 2]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([1.9, 0.9, 0.4]), 
                    fixed_cons=np.array([2, 2, 2]), 
                    battery_start=1.1)
        conf = Conf(battery_min=0.1,
                    battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([-1, 0, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_battery_plus_fixed_cons_zero_sell_price(self):

        data = Data(sol=np.array([4, 1, 10]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([0, 0, 0]), 
                    fixed_cons=np.array([2, 2, 2]), 
                    battery_start=0)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([2, -1, 8]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_eq_battery_plus_fixed_cons_plus_sell(self):

        data = Data(sol=np.array([12, 1, 10]), 
                    grid_buy=np.array([1.0, 2.0, 0.5]), 
                    grid_sell=np.array([0.9, 1.9, 0.4]), 
                    fixed_cons=np.array([2, 3, 2]), 
                    battery_start=10)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([10, -20, 0]),
                              buy=np.array([0, 0, 0]),
                              sell=np.array([0, 18, 8]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_sol_plus_buy_eq_battery_plus_fixed_cons_plus_sell(self):

        data = Data(sol=np.array([8, 1, 10]), 
                    grid_buy=np.array([1.0, 2.0, 0.5]), 
                    grid_sell=np.array([0.9, 1.9, 0.4]), 
                    fixed_cons=np.array([2, 3, 2]), 
                    battery_start=10)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([10, -20, 0]),
                              buy=np.array([4, 0, 0]),
                              sell=np.array([0, 18, 8]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_buy_low_sell_high_with_battery(self):

        data = Data(sol=np.array([0, 0]), 
                    grid_buy=np.array([1.0, 2.0]), 
                    grid_sell=np.array([0.9, 1.9]), 
                    fixed_cons=np.array([0, 0]), 
                    battery_start=0.5)
        conf = Conf(battery_min=0.1, 
                    battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([19.5, -19.9]),
                              buy=np.array([19.5, 0]),
                              sell=np.array([0, 19.9]),
                              cons=np.array([0, 0]))        
        assert_result(res, expected_res)
    
    def test_buy_low_sell_high_with_sol_and_fixed_cons(self):

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

    def test_sol_plus_buy_eq_fixed_cons_plus_sell(self):

        data = Data(sol=np.array([4, 1, 3]), 
                    grid_buy=np.array([2.0, 1.0, 0.5]), 
                    grid_sell=np.array([2.0, 1.0, 0.5]), 
                    fixed_cons=np.array([2.5, 1.4, 3.2]), 
                    battery_start=0)
        conf = Conf(battery_max=20)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([0, 0, 0]),
                              buy=np.array([0, 0.4, 0.2]),
                              sell=np.array([1.5, 0, 0]),
                              cons=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_neg_grid_buy(self):

        data = Data(sol=np.array([2.0, 1.0]), 
                    grid_buy=np.array([1.0, -2.0]), 
                    grid_sell=np.array([1.0, -2.0]), 
                    fixed_cons=np.array([2.5, 1.4]), 
                    battery_start=0.5)
        conf = Conf(battery_max=2,
                    buy_max=10)

        res = optimize(data, conf)

        expected_res = Result(battery=np.array([-0.5, 2]),
                              buy=np.array([0, 10]),
                              sell=np.array([0, 0]),
                              cons=np.array([0, 7.6]))        
        assert_result(res, expected_res)


def assert_result(res, expected_res):
        np.testing.assert_array_almost_equal(res.battery, expected_res.battery)
        np.testing.assert_array_almost_equal(res.buy, expected_res.buy)
        np.testing.assert_array_almost_equal(res.sell, expected_res.sell)
        np.testing.assert_array_almost_equal(res.cons, expected_res.cons)
    

if __name__ == '__main__':
    unittest.main()
