"""Offline unit tests for the financial math in portfolio.py.

No network: everything runs on synthetic data. Run with
    python3 -m unittest discover -s tests
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import portfolio  # noqa: E402


def _instruments():
    return pd.DataFrame([
        {"ticker": "XEQT", "yf_symbol": "XEQT.TO", "currency": "CAD",
         "is_private": False, "manual_price": None, "last_price": None},
        {"ticker": "XUS", "yf_symbol": "XUS.TO", "currency": "CAD",
         "is_private": False, "manual_price": None, "last_price": None},
        {"ticker": "AVGE", "yf_symbol": "AVGE", "currency": "USD",
         "is_private": False, "manual_price": None, "last_price": None},
        {"ticker": "OPO", "yf_symbol": None, "currency": "CAD",
         "is_private": True, "manual_price": 17.184, "last_price": None},
    ])


def _positions():
    return pd.DataFrame([
        {"Ticker": "XEQT", "Market Value": 14000.0},
        {"Ticker": "XUS", "Market Value": 10500.0},
        {"Ticker": "AVGE", "Market Value": 2200.0},
        {"Ticker": "OPO", "Market Value": 10800.0},
    ])


class TestMaxDrawdown(unittest.TestCase):
    def test_known_drawdown(self):
        s = pd.Series([100, 120, 60, 90, 130])
        self.assertAlmostEqual(portfolio.max_drawdown(s), -0.5)

    def test_monotone_series_has_zero_drawdown(self):
        s = pd.Series([100, 110, 120, 130])
        self.assertAlmostEqual(portfolio.max_drawdown(s), 0.0)

    def test_short_series_returns_none(self):
        self.assertIsNone(portfolio.max_drawdown(pd.Series([100.0])))


class TestXirr(unittest.TestCase):
    def test_doubling_in_one_year(self):
        # ACT/365.25 day count: 365 days is fractionally under a year, so the
        # solved rate sits a hair above 1.0 — assert to 2 places.
        flows = [(dt.date(2025, 1, 1), -100.0), (dt.date(2026, 1, 1), 200.0)]
        self.assertAlmostEqual(portfolio.xirr(flows), 1.0, places=2)

    def test_flat_is_zero(self):
        flows = [(dt.date(2025, 1, 1), -100.0), (dt.date(2026, 1, 1), 100.0)]
        self.assertAlmostEqual(portfolio.xirr(flows), 0.0, places=4)

    def test_same_sign_returns_none(self):
        flows = [(dt.date(2025, 1, 1), -100.0), (dt.date(2026, 1, 1), -100.0)]
        self.assertIsNone(portfolio.xirr(flows))

    def test_multi_flow_direction(self):
        flows = [(dt.date(2025, 1, 1), -100.0), (dt.date(2025, 7, 1), -100.0),
                 (dt.date(2026, 1, 1), 230.0)]
        r = portfolio.xirr(flows)
        self.assertGreater(r, 0.15)
        self.assertLess(r, 0.35)


class TestTwr(unittest.TestCase):
    def test_chain_linking_ignores_flows(self):
        snaps = pd.DataFrame({
            "date": [dt.date(2026, 7, 1), dt.date(2026, 7, 2), dt.date(2026, 7, 3)],
            "daily_pnl_pct": [0.01, -0.02, 0.03],
        })
        twr = portfolio.twr_series(snaps)
        expected = 100 * 1.01 * 0.98 * 1.03
        self.assertAlmostEqual(float(twr.iloc[-1]), expected, places=6)

    def test_empty(self):
        self.assertTrue(portfolio.twr_series(pd.DataFrame()).empty)


class TestLookThrough(unittest.TestCase):
    def test_weights_sum_to_one_and_opo_excluded(self):
        lt = portfolio.look_through(_positions(), _instruments())
        self.assertAlmostEqual(sum(lt["blocks"].values()), 1.0, places=9)
        self.assertAlmostEqual(sum(lt["region"].values()), 1.0, places=9)
        self.assertNotIn("OPO", lt["blocks"])

    def test_xus_is_us_market(self):
        pos = pd.DataFrame([{"Ticker": "XUS", "Market Value": 100.0}])
        lt = portfolio.look_through(pos, _instruments())
        self.assertAlmostEqual(lt["blocks"]["US Market"], 1.0)


class TestPolicy(unittest.TestCase):
    def test_policy_sleeves_sum_to_one(self):
        self.assertAlmostEqual(sum(portfolio.POLICY_SLEEVES.values()), 1.0, places=9)

    def test_policy_benchmark_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(portfolio.POLICY_BENCHMARK.values()), 1.0,
                               places=9)

    def test_bl_prior_sums_to_one(self):
        self.assertAlmostEqual(sum(portfolio.BL_PRIOR.values()), 1.0, places=9)


class TestNextDollar(unittest.TestCase):
    def test_allocation_sums_to_amount_and_never_sells(self):
        ndf = portfolio.next_dollar(_positions(), _instruments(), 1000.0)
        self.assertFalse(ndf.empty)
        self.assertAlmostEqual(float(ndf["buy"].sum()), 1000.0, places=6)
        self.assertTrue((ndf["buy"] >= 0).all())

    def test_underweights_get_funded_first(self):
        ndf = portfolio.next_dollar(_positions(), _instruments(), 1000.0)
        # AVUS is heavily overweight vs the 35% policy: no new dollars to it
        self.assertNotIn("AVUS", set(ndf["sleeve"]))
        self.assertIn("AVIV", set(ndf["sleeve"]))

    def test_zero_amount(self):
        self.assertTrue(
            portfolio.next_dollar(_positions(), _instruments(), 0.0).empty)


class TestContributionRoom(unittest.TestCase):
    def test_tfsa_room_from_age_18(self):
        # born 2002 → room accrues 2020 onward
        expected = sum(v for y, v in
                       __import__("db").TFSA_ANNUAL_LIMITS.items() if y >= 2020)
        self.assertAlmostEqual(portfolio.tfsa_cumulative_room(2026), expected)

    def test_fhsa_carryforward_caps_at_8000(self):
        st = portfolio.fhsa_status({2025: 0.0}, 2026)
        self.assertAlmostEqual(st["available_this_year"], 16000.0)


class TestFxHelpers(unittest.TestCase):
    def test_symbol_currency_heuristic(self):
        self.assertTrue(portfolio._is_cad("XEQT.TO"))
        self.assertFalse(portfolio._is_cad("AVUS"))
        self.assertFalse(portfolio._is_cad("^GSPC"))

    def test_money_weighted_return_math(self):
        contribs = pd.DataFrame({
            "date": [dt.date(2025, 7, 10)], "amount": [100.0]})
        mw = portfolio.money_weighted_return(contribs, 110.0)
        self.assertAlmostEqual(mw["contributed"], 100.0)
        self.assertAlmostEqual(mw["growth"], 10.0)
        self.assertIsNotNone(mw["xirr"])


if __name__ == "__main__":
    unittest.main()
