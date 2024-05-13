import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
import logging
from datetime import datetime

log_filename = f"/Users/arbili/Arik/CubicForwardSpline/log/CubicForwardSpline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename)
logger = logging.getLogger('CubicForwardSpline')

class CubicForwardSpline:
    """
    A class for calculating cubic forward spline interpolation.

    Args:
        in_prices (pandas.DataFrame): Input prices data.
        in_cf (pandas.DataFrame): Input cash flow data.
        in_params (pandas.DataFrame): Input parameters for the spline.

    Attributes:
        prices (pandas.DataFrame): Input prices data.
        cf (pandas.DataFrame): Input cash flow data.
        params (pandas.DataFrame): Input parameters for the spline.
        adj_params_flatten (numpy.ndarray): Flattened adjusted parameters.
        run_cubic (bool): Flag indicating whether to run cubic spline or linear spline.

    Methods:
        calc_spline(in_adj_params_flatten): Calculates the spline interpolation.
        run_one_time(): Runs the spline interpolation once without optimization.
        run_linear(): Runs the linear spline optimization.
        run_cubic_spline(): Runs the cubic spline optimization.
        load_data(in_prices, in_cf, in_params): Loads data from files.
    """

    def __init__(self, in_prices=None, in_cf=None, in_params=None):
        """
        Initializes the CubicForwardSpline class.

        Args:
            in_prices (pandas.DataFrame): Input prices data.
            in_cf (pandas.DataFrame): Input cash flow data.
            in_params (pandas.DataFrame): Input parameters for the spline.
        """
        logger.info("Initializing CubicForwardSpline")
        try:
            self.prices = in_prices
            self.cf = in_cf
            self.params = in_params
            if in_params is not None:
                self.adj_params = in_params[['b0', 'b1', 'b2', 'b3']].copy()
                self.adj_params_flatten = self.adj_params.values.flatten()

            if in_cf is not None:
                self.cf['pdate'] = pd.to_datetime(self.cf['pdate'], dayfirst=True)
                self.cf['tdate'] = pd.to_datetime(self.cf['tdate'], dayfirst=True)
            self.run_cubic = True
            self.adj_params_flatten = None
            logger.info("Finished initializing CubicForwardSpline")
        except Exception as e:
            logger.exception("Exception occurred while initializing CubicForwardSpline")
            raise e

    def calc_spline(self, in_adj_params_flatten):
        """
        Calculates the spline interpolation.

        Args:
            in_adj_params_flatten (numpy.ndarray): Flattened adjusted parameters.

        Returns:
            float: The sum of absolute errors between calculated prices and actual prices.
        """
        logger.info("Starting calc_spline")
        try:
            adj_params_reconstructed = pd.DataFrame(in_adj_params_flatten.reshape(self.adj_params.shape), columns=self.adj_params.columns)

            if self.run_cubic:
                self.params['b0'] = adj_params_reconstructed['b0']
                self.params['b1'] = adj_params_reconstructed['b1']
                self.params['b2'] = adj_params_reconstructed['b2']
                self.params['b3'] = adj_params_reconstructed['b3']
            else:
                self.params['b0'] = adj_params_reconstructed['b0']
                self.params['b1'] = adj_params_reconstructed['b1']
                self.params['b2'] = 0
                self.params['b3'] = 0

            # Update params
            self.params['f1'] = self.params['b0'] + self.params['b1'] * self.params['t1'] + self.params['b2'] * self.params['t1'] * self.params['t1'] + \
                                self.params['b2'] * self.params['t1'] * self.params['t1'] * self.params['t1']
            self.params['f2'] = self.params['b0'] + self.params['b1'] * self.params['t2'] + self.params['b2'] * self.params['t2'] * self.params['t2'] + \
                                self.params['b2'] * self.params['t2'] * self.params['t2'] * self.params['t2']

            self.params['f1'] = self.params['f2'].shift(1)

            self.params['int1'] = self.params['b0'] * self.params['t1'] + \
                                  self.params['b1'] * self.params['t1'] * self.params['t1'] * 0.5 + \
                                  self.params['b2'] * self.params['t1'] * self.params['t1'] * self.params['t1'] * 0.333333 + \
                                  self.params['b3'] * self.params['t1'] * self.params['t1'] * self.params['t1'] * self.params['t1'] * 0.25

            self.params['int2'] = self.params['b0'] * self.params['t2'] + \
                                  self.params['b1'] * self.params['t2'] * self.params['t2'] * 0.5 + \
                                  self.params['b2'] * self.params['t2'] * self.params['t2'] * self.params['t2'] * 0.333333 + \
                                  self.params['b3'] * self.params['t2'] * self.params['t2'] * self.params['t2'] * self.params['t2'] * 0.25

            self.params['space'] = self.params['int2'] - self.params['int1']
            self.params['acc_space'] = self.params['space'].shift(1).cumsum().fillna(0)
            self.params['f_tag'] = self.params['b1'] + 2 * self.params['b2'] * self.params['t1'] + 3 * self.params['b3'] * self.params['t1'] * self.params['t1'].fillna(0)
            self.params['f_tag_square'] = (self.params['f_tag'] - self.params['f_tag'].shift(1)).fillna(0)
            self.params['wgt_tag'] = self.params['f_tag_square'] * self.params['wgt']
            params_cols_to_keep = ['t1', 't2', 'b0', 'b1', 'b2', 'b3', 'acc_space']
            
            # CF section
            self.cf['mat'] = ((self.cf['pdate'] - self.cf['tdate']).dt.days - 1) / 365
            cf_new = self.cf.groupby('ncode', group_keys=False).apply(
                lambda x: pd.merge_asof(x, self.params[params_cols_to_keep], left_on='mat', right_on='t2', direction='forward'))

            cf_new['f'] = cf_new['b0'] + cf_new['b1'] * cf_new['t1']
            cf_new['int1'] = cf_new['b0'] * cf_new['t1'] + \
                             cf_new['b1'] * cf_new['t1'] * cf_new['t1'] * 0.5 + \
                             cf_new['b2'] * cf_new['t1'] * cf_new['t1'] * cf_new['t1'] * 0.333333 + \
                             cf_new['b3'] * cf_new['t1'] * cf_new['t1'] * cf_new['t1'] * cf_new['t1'] * 0.25

            cf_new['int2'] = cf_new['b0'] * cf_new['mat'] + \
                             cf_new['b1'] * cf_new['mat'] * cf_new['mat'] * 0.5 + \
                             cf_new['b2'] * cf_new['mat'] * cf_new['mat'] * cf_new['mat'] * 0.333333 + \
                             cf_new['b3'] * cf_new['mat'] * cf_new['mat'] * cf_new['mat'] * cf_new['t1'] * 0.25

            cf_new['space'] = cf_new['int2'] - cf_new['int1']
            cf_new['r'] = (cf_new['acc_space'] + cf_new['space']) / cf_new['mat']
            cf_new['pv'] = cf_new['pmt'] / np.power((1 + cf_new['r']), cf_new['mat'])
            calc_prices = cf_new.groupby('ncode')['pv'].sum()
            error_calc_df = self.prices.merge(calc_prices, on='ncode')
            ret = np.abs(error_calc_df['price'] - error_calc_df['pv']).sum()
            logger.info("Finished calc_spline")
            return ret
        except Exception as e:
            logger.exception("Exception occurred in calc_spline")
            raise e

    def run_one_time(self):
        """
        Runs the spline interpolation once without optimization.

        Returns:
            float: The sum of absolute errors between calculated prices and actual prices.
        """
        logger.info("Starting run_one_time")
        try:
            self.run_cubic = False
            result = self.calc_spline(self.adj_params_flatten)
            logger.info("Finished run_one_time")
            return result
        except Exception as e:
            logger.exception("Exception occurred in run_one_time")
            raise e

    def run_linear(self):
        """
        Runs the linear spline optimization.

        Returns:
            scipy.optimize.OptimizeResult: The optimization result.
        """
        logger.info("Starting run_linear")
        try:
            self.run_cubic = False
            ret = minimize(fun=self.calc_spline, x0=self.adj_params_flatten, method='L-BFGS-B', options={'ftol': 1e-02, 'maxiter': 1000})
            logger.info("Finished run_linear")
            return ret
        except Exception as e:
            logger.exception("Exception occurred in run_linear")
            raise e

    def run_cubic_spline(self):
        """
        Runs the cubic spline optimization.

        Returns:
            scipy.optimize.OptimizeResult: The optimization result.
        """
        logger.info("Starting run_cubic_spline")
        try:
            self.run_cubic = True
            ret = minimize(fun=self.calc_spline, x0=self.adj_params_flatten, method='L-BFGS-B', options={'ftol': 1e-02, 'maxiter': 10000})
            logger.info("Finished run_cubic_spline")
            return ret
        except Exception as e:
            logger.exception("Exception occurred in run_cubic_spline")
            raise e

    def load_data(self, in_prices, in_cf, in_params):
        """
        Loads data from files.

        Args:
            in_prices (str): Path to the prices CSV file.
            in_cf (str): Path to the cash flow CSV file.
            in_params (str): Path to the parameters CSV file.
        """
        logger.info("Starting load_data")
        try:
            self.prices = pd.read_csv(in_prices)
            self.cf = pd.read_csv(in_cf)
            self.params = pd.read_csv(in_params)
            self.adj_params = self.params[['b0', 'b1', 'b2', 'b3']].copy()
            self.adj_params_flatten = self.adj_params.values.flatten()

            if self.cf is not None:
                self.cf['pdate'] = pd.to_datetime(self.cf['pdate'], dayfirst=True)
                self.cf['tdate'] = pd.to_datetime(self.cf['tdate'], dayfirst=True)
            logger.info("Finished load_data")
        except Exception as e:
            logger.exception("Exception occurred in load_data")
            raise e