import numpy as np
import unittest
from optimizer import optimize
from data import Data
from conf import Conf
from result import Result

class test_hema(unittest.TestCase):

    def test_import_power_eq_baseline_load_power(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([2.0, 1.0, 0.5]), 
                    baseline_load_power=np.array([2.5, 1.4, 3.2]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0]),
                              import_power=np.array([2.5, 1.4, 3.2]),
                              export_power=np.array([0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_eq_export_power_limited_import_power_battery_max(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0]), 
                    grid_import_price=np.array([1.0, 1.1, 0.9]), 
                    grid_export_price=np.array([1.0, 1.1, 0.9]), 
                    baseline_load_power=np.array([0, 0, 0]), 
                    battery_current_soc=1.0)
        conf = Conf(battery_capacity=10)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, -10, 0]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 10, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_eq_export_power(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0]), 
                    grid_import_price=np.array([1.1, 1.2, 0.9]), 
                    grid_export_price=np.array([1.0, 1.1, 0.9]), 
                    baseline_load_power=np.array([0, 0, 0]), 
                    battery_current_soc=0.5)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, -10, 0]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 10, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_plus_import_power_eq_export_power(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0]), 
                    grid_import_price=np.array([1.0, 1.1, 0.9]), 
                    grid_export_price=np.array([1.0, 1.1, 0.9]), 
                    baseline_load_power=np.array([0, 0, 0]), 
                    battery_current_soc=0.4)
        conf = Conf(battery_capacity=25,
                    battery_max_soc=0.8)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([10, -20, 0]),
                              import_power=np.array([10, 0, 0]),
                              export_power=np.array([0, 20, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_plus_import_power_eq_export_power_with_limited_charging_power(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0, 0, 0]), 
                    grid_import_price=np.array([1.0, 1.2, 0.9, 1.1, 1.3]), 
                    grid_export_price=np.array([1.0, 1.2, 0.9, 1.1, 1.3]), 
                    baseline_load_power=np.array([0, 0, 0, 0, 0]), 
                    battery_current_soc=0.2)
        conf = Conf(battery_capacity=20,
                    battery_max_soc=0.95,
                    battery_charge_power=5,
                    battery_discharge_power=6)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([5, -6, 5, -2, -6]),
                              import_power=np.array([5, 0, 5, 0, 0]),
                              export_power=np.array([0, 6, 0, 2, 6]),
                              controllable_load_power=np.array([0, 0, 0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_eq_baseline_load_power(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0]), 
                    grid_import_price=np.array([1.1, 2.2, 1.5]), 
                    grid_export_price=np.array([0, 0, 0]), 
                    baseline_load_power=np.array([2, 3, 5]), 
                    battery_current_soc=0.5)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([-2, -3, -5]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_eq_baseline_load_power_plus_export_power(self):
        data = Data(pv_forecast_power=np.array([0, 0, 0]), 
                    grid_import_price=np.array([1.2, 1.3, 1.2]), 
                    grid_export_price=np.array([1.0, 1.1, 0.9]), 
                    baseline_load_power=np.array([2, 2, 4]), 
                    battery_current_soc=0.5)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([-2, -4, -4]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 2, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_baseline_load_power(self):
        data = Data(pv_forecast_power=np.array([4, 1, 3]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([2.0, 1.0, 0.5]), 
                    baseline_load_power=np.array([4, 1, 3]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_export_power(self):
        data = Data(pv_forecast_power=np.array([4, 1, 3]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([2.0, 1.0, 0.5]), 
                    baseline_load_power=np.array([0, 0, 0]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([4, 1, 3]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_baseline_load_power_plus_export_power(self):
        data = Data(pv_forecast_power=np.array([4, 1, 3.1]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([2.0, 1.0, 0.5]), 
                    baseline_load_power=np.array([2.5, 1, 3]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([1.5, 0, 0.1]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power(self):
        data = Data(pv_forecast_power=np.array([4, 1, 3]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([0, 0, 0]), 
                    baseline_load_power=np.array([0, 0, 0]), 
                    battery_current_soc=0)
        conf = Conf(battery_max_soc=1.0,
                    battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([4, 1, 3]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power_plus_export_power(self):
        data = Data(pv_forecast_power=np.array([4, 1, 10]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([1.9, 0.9, 0.5]), 
                    baseline_load_power=np.array([0, 0, 0]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20,
                    export_max_power=3.5)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0.5, -0.5, 6.5]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([3.5, 1.5, 3.5]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power_plus_baseline_load_power(self):
        data = Data(pv_forecast_power=np.array([1, 2, 2]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([1.9, 0.9, 0.4]), 
                    baseline_load_power=np.array([2, 2, 2]), 
                    battery_current_soc=0.055)
        conf = Conf(battery_min_soc=0.005,
                    battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([-1, 0, 0]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power_plus_baseline_load_power_zero_export_price(self):
        data = Data(pv_forecast_power=np.array([4, 1, 10]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([0, 0, 0]), 
                    baseline_load_power=np.array([2, 2, 2]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([2, -1, 8]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power_plus_baseline_load_power_plus_export_power(self):
        data = Data(pv_forecast_power=np.array([12, 1, 10]), 
                    grid_import_price=np.array([0.5, 2.0, 1.0]), 
                    grid_export_price=np.array([0.4, 1.9, 0.9]), 
                    baseline_load_power=np.array([2, 3, 2]), 
                    battery_current_soc=0.08)
        conf = Conf(battery_capacity=100,
                    battery_charge_power=9,
                    battery_discharge_power=9)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([9, -9, -8]),
                              import_power=np.array([0, 0, 0]),
                              export_power=np.array([1, 7, 16]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_plus_import_power_eq_battery_power_plus_baseline_load_power_plus_export_power(self):
        data = Data(pv_forecast_power=np.array([8, 1, 10]), 
                    grid_import_price=np.array([1.0, 2.0, 0.5]), 
                    grid_export_price=np.array([0.9, 1.9, 0.4]), 
                    baseline_load_power=np.array([2, 3, 2]), 
                    battery_current_soc=0.5)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([10, -20, 0]),
                              import_power=np.array([4, 0, 0]),
                              export_power=np.array([0, 18, 8]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_buy_low_sell_high_with_battery(self):
        data = Data(pv_forecast_power=np.array([0, 0]), 
                    grid_import_price=np.array([1.0, 2.0]), 
                    grid_export_price=np.array([0.9, 1.9]), 
                    baseline_load_power=np.array([0, 0]), 
                    battery_current_soc=0.025)
        conf = Conf(battery_min_soc=0.005, 
                    battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([19.5, -19.9]),
                              import_power=np.array([19.5, 0]),
                              export_power=np.array([0, 19.9]),
                              controllable_load_power=np.array([0, 0]),
                              excess_load_power=np.array([0, 0]))        
        assert_result(res, expected_res)
    
    def test_buy_low_sell_high_with_pv_forecast_power_and_baseline_load_power(self):
        data = Data(pv_forecast_power=np.array([2.0, 1.0]), 
                    grid_import_price=np.array([1.0, 2.0]), 
                    grid_export_price=np.array([0.9, 1.9]), 
                    baseline_load_power=np.array([0.5, 0.4]), 
                    battery_current_soc=0.025)
        conf = Conf(battery_min_soc=0.005, 
                    battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([19.5, -19.9]),
                              import_power=np.array([18, 0]),
                              export_power=np.array([0, 20.5]),
                              controllable_load_power=np.array([0, 0]),
                              excess_load_power=np.array([0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_plus_import_power_eq_baseline_load_power_plus_export_power(self):
        data = Data(pv_forecast_power=np.array([4, 1, 3]), 
                    grid_import_price=np.array([2.0, 1.0, 0.5]), 
                    grid_export_price=np.array([2.0, 1.0, 0.5]), 
                    baseline_load_power=np.array([2.5, 1.4, 3.2]), 
                    battery_current_soc=0)
        conf = Conf(battery_capacity=20)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0]),
                              import_power=np.array([0, 0.4, 0.2]),
                              export_power=np.array([1.5, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0]))        
        assert_result(res, expected_res)

    def test_neg_grid_import_price(self):
        data = Data(pv_forecast_power=np.array([2.0, 1.0]), 
                    grid_import_price=np.array([1.0, -2.0]), 
                    grid_export_price=np.array([1.0, -2.0]), 
                    baseline_load_power=np.array([2.5, 1.4]), 
                    battery_current_soc=0.25)
        conf = Conf(battery_capacity=2,
                    import_max_power=10)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([-0.5, 2]),
                              import_power=np.array([0, 10]),
                              export_power=np.array([0, 0]),
                              controllable_load_power=np.array([0, 0]),
                              excess_load_power=np.array([0, 7.6]))        
        assert_result(res, expected_res)

    def test_controllable_load_power_off_at_high_import_power_max_gap_2(self):
        data = Data(grid_import_price=np.array([2.0, 1.0, 1.1, 0.5, 1.2, 2.3, 1.1]), 
                    grid_export_price=np.array([2.0, 1.0, 1.1, 0.5, 1.2, 2.3, 1.1]),
                    controllable_load_power=np.array([1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]))
        conf = Conf(battery_capacity=0,
                    max_hours_gap=2,
                    max_hours=4)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0, 0, 0, 0, 0]),
                              import_power=np.array([0, 1.7, 1.7, 1.7, 0, 0, 1.7]),
                              export_power=np.array([0, 0, 0, 0, 0, 0, 0]),
                              controllable_load_power=np.array([0, 1.7, 1.7, 1.7, 0, 0, 1.7]),
                              excess_load_power=np.array([0, 0, 0, 0, 0, 0, 0]))        
        assert_result(res, expected_res)

    def test_controllable_load_power_off_at_high_import_power_max_gap_1(self):
        data = Data(grid_import_price=np.array([1.6, 1.5, 1.6, 1.5, 1.1, 2.3, 1.3]), 
                    grid_export_price=np.array([1.6, 1.5, 1.6, 1.5, 1.1, 2.3, 1.3]), 
                    controllable_load_power=np.array([1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]))
        conf = Conf(battery_capacity=0,
                    max_hours_gap=1,
                    max_hours=4)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([0, 0, 0, 0, 0, 0, 0]),
                              import_power=np.array([0, 1.7, 0, 1.7, 1.7, 0, 1.7]),
                              export_power=np.array([0, 0, 0, 0, 0, 0, 0]),
                              controllable_load_power=np.array([0, 1.7, 0, 1.7, 1.7, 0, 1.7]),
                              excess_load_power=np.array([0, 0, 0, 0, 0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power_with_battery_energy_loss(self):
        data = Data(pv_forecast_power=np.array([10, 15, 21.1, 0]),
                    grid_import_price=np.array([1, 1, 1, 1]))
        conf = Conf(battery_energy_loss=0.08)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([9.2, 13.8, 19.412, 0]),
                              import_power=np.array([0, 0, 0, 0]),
                              export_power=np.array([0, 0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0, 0]))        
        assert_result(res, expected_res)

    def test_pv_forecast_power_eq_battery_power_with_pv_energy_loss(self):
        data = Data(pv_forecast_power=np.array([10, 15, 21.1, 0]),
                    grid_import_price=np.array([1, 1, 1, 1]))
        conf = Conf(pv_energy_loss=0.08)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([9.2, 13.8, 19.412, 0]),
                              import_power=np.array([0, 0, 0, 0]),
                              export_power=np.array([0, 0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0, 0]))        
        assert_result(res, expected_res)

    def test_battery_power_eq_baseline_load_power_with_battery_energy_loss(self):
        data = Data(grid_import_price=np.array([1, 1, 1, 1]),
                    baseline_load_power=np.array([1.5, 2.66, 2, 0]),
                    battery_current_soc=0.1)
        conf = Conf(battery_capacity=100,
                    battery_discharge_power=2.5,
                    battery_energy_loss=0.06)

        res = optimize(data, conf)

        expected_res = Result(battery_power=np.array([-1.595745, -2.5, -2.12766, 0]),
                              import_power=np.array([0, 0.31, 0, 0]),
                              export_power=np.array([0, 0, 0, 0]),
                              controllable_load_power=np.array([0, 0, 0, 0]),
                              excess_load_power=np.array([0, 0, 0, 0]))        
        assert_result(res, expected_res)

def assert_result(res, expected_res):
        np.testing.assert_array_almost_equal(res.battery_power, expected_res.battery_power)
        np.testing.assert_array_almost_equal(res.import_power, expected_res.import_power)
        np.testing.assert_array_almost_equal(res.export_power, expected_res.export_power)
        np.testing.assert_array_almost_equal(res.controllable_load_power, expected_res.controllable_load_power)
        np.testing.assert_array_almost_equal(res.excess_load_power, expected_res.excess_load_power)
    

if __name__ == '__main__':
    unittest.main()
