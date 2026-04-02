"""Comprehensive test for Greeks, VaR, and Risk modules."""
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_greeks():
    """Test Greeks calculator."""
    print("\n" + "="*60)
    print("Testing Greeks Calculator")
    print("="*60)
    
    try:
        from src.models.greeks import (
            GreeksCalculator,
            portfolio_greeks,
            delta_neutral_hedge,
            gamma_exposure_metric,
            vega_exposure_metric,
        )
        
        calc = GreeksCalculator()
        
        # Standard option parameters
        S = 100      # spot
        K = 100      # strike (ATM)
        T = 30/365   # 30 days to maturity
        r = 0.03     # 3% risk-free rate
        sigma = 0.20 # 20% volatility
        
        # Calculate Greeks
        delta_call = calc.delta(S, K, T, r, sigma, "call")
        delta_put = calc.delta(S, K, T, r, sigma, "put")
        gamma = calc.gamma(S, K, T, r, sigma)
        vega = calc.vega(S, K, T, r, sigma)
        theta = calc.theta(S, K, T, r, sigma, "call")
        rho = calc.rho(S, K, T, r, sigma, "call")
        
        print(f"✓ Greeks computed successfully:")
        print(f"  - Call Delta: {delta_call:.4f}")
        print(f"  - Put Delta: {delta_put:.4f}")
        print(f"  - Gamma: {gamma:.6f}")
        print(f"  - Vega: {vega:.2f} (per 1% vol)")
        print(f"  - Theta: {theta:.6f} (daily)")
        print(f"  - Rho: {rho:.4f} (per 1% rate)")
        
        # Test portfolio Greeks
        positions = [
            {"quantity": 10, "S": S, "K": K-5, "T": T, "r": r, "sigma": sigma, "type": "call"},
            {"quantity": -10, "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "type": "put"},
        ]
        port_greeks = portfolio_greeks(positions)
        print(f"\n✓ Portfolio Greeks (10 calls - 10 puts):")
        print(f"  - Delta: {port_greeks['delta']:.4f}")
        print(f"  - Gamma: {port_greeks['gamma']:.6f}")
        print(f"  - Vega: {port_greeks['vega']:.2f}")
        
        # Test delta hedge
        hedge_shares = delta_neutral_hedge(
            spot_price=S,
            position_delta=port_greeks['delta'],
            hedge_instrument_delta=1.0,
        )
        print(f"\n✓ Delta Hedge Size: {hedge_shares:.2f} shares to sell")
        
        # Test gamma exposure
        gamma_pnl = gamma_exposure_metric(gamma, S, move_pct=0.02)
        print(f"\n✓ Gamma P&L under 2% move: ${gamma_pnl:.2f}")
        
        # Test vega exposure
        vega_pnl = vega_exposure_metric(vega, vol_move_pct=0.05)
        print(f"✓ Vega P&L under +5% vol move: ${vega_pnl:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_monte_carlo_var():
    """Test Monte Carlo VaR/ES."""
    print("\n" + "="*60)
    print("Testing Monte Carlo VaR/ES")
    print("="*60)
    
    try:
        from src.risk.monte_carlo_var import MonteCarloVaR, stress_test_portfolio
        
        mc = MonteCarloVaR()
        
        # Single asset 1-day VaR
        var_result = mc.one_day_var(
            current_price=100.0,
            position_size=1000,  # 1000 shares
            expected_return=0.0005,  # daily
            volatility=0.01,  # daily
            confidence=0.99,
            num_paths=10000,
            seed=42,
        )
        
        print(f"✓ Single-asset 1-day VaR (1000 shares @ $100):")
        print(f"  - 99% VaR Loss: ${var_result['var_loss']:,.2f}")
        print(f"  - 99% ES Loss: ${var_result['es_loss']:,.2f}")
        print(f"  - VaR Loss %: {var_result['var_loss_pct']:.2f}%")
        
        # Portfolio VaR
        holdings = {
            'AAPL': {
                'price': 150,
                'shares': 100,
                'daily_return': 0.0008,
                'daily_vol': 0.018,
            },
            'MSFT': {
                'price': 300,
                'shares': 50,
                'daily_return': 0.0007,
                'daily_vol': 0.016,
            },
        }
        
        port_var = mc.portfolio_var(
            holdings=holdings,
            confidence=0.99,
            num_paths=10000,
            seed=42,
        )
        
        port_value = sum(h['price'] * h['shares'] for h in holdings.values())
        print(f"\n✓ Portfolio VaR (2-stock portfolio worth ${port_value:,.0f}):")
        print(f"  - 99% VaR Loss: ${port_var['portfolio_var']:,.2f}")
        print(f"  - 99% ES Loss: ${port_var['portfolio_es']:,.2f}")
        print(f"  - VaR Loss %: {port_var['portfolio_var_pct']:.2f}%")
        print(f"  - Simulation paths: {port_var['num_simulations']}")
        
        # Stress testing
        scenarios = {
            'market_crash_10pct': -0.10,
            'tech_surge_5pct': 0.05,
            'vol_spike': 0.0,  # not applied to price directly
        }
        stress_results = stress_test_portfolio(holdings, scenarios)
        print(f"\n✓ Stress Test Results:")
        for scenario, result in stress_results.items():
            print(f"  - {scenario}: P&L = ${result['total_pnl']:,.2f} ({result['pnl_pct']:.2f}%)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_risk_manager():
    """Test integrated risk manager."""
    print("\n" + "="*60)
    print("Testing Risk Manager")
    print("="*60)
    
    try:
        from src.risk.risk_manager import RiskManager, rebalance_greeks
        
        rm = RiskManager(
            initial_capital=100000,
            max_var_loss_pct=2.0,
            max_delta_notional_pct=1.0,
            enable_delta_hedge=True,
            enable_var_limit=True,
        )
        
        print(f"✓ Risk Manager initialized:")
        print(f"  - Capital: ${rm.initial_capital:,.0f}")
        print(f"  - Max VaR Loss: {rm.max_var_loss_pct}%")
        print(f"  - Max Delta Notional: {rm.max_delta_notional_pct}%")
        
        # Test option position
        position = {
            "quantity": 10,
            "S": 100,
            "K": 105,
            "T": 30/365,
            "r": 0.03,
            "sigma": 0.20,
            "type": "call",
        }
        rm.add_option_position(position)
        print(f"\n✓ Added option position (10 calls)")
        
        # Compute Greeks
        greeks = rm.compute_greeks()
        print(f"✓ Portfolio Greeks:")
        print(f"  - Delta: {greeks['delta']:.4f}")
        print(f"  - Gamma: {greeks['gamma']:.6f}")
        print(f"  - Vega: {greeks['vega']:.2f}")
        
        # Test delta hedge
        hedge_size = rm.suggest_delta_hedge_size(100)
        print(f"\n✓ Suggested delta hedge: {hedge_size:.2f} shares")
        
        # Get risk summary
        risk_summary = rm.get_risk_summary()
        print(f"\n✓ Risk Summary:")
        print(f"  - Greeks: {risk_summary['greeks']}")
        print(f"  - Max VaR Loss: {risk_summary['max_var_loss_pct']}%")
        print(f"  - Delta hedge enabled: {risk_summary['enable_delta_hedge']}")
        
        # Test rebalancing decision
        current_greeks = {"delta": 50, "gamma": 0.1, "vega": 200}
        target_greeks = {"delta": 0, "gamma": 0, "vega": 0}
        rebal = rebalance_greeks(current_greeks, target_greeks, rebalance_threshold=0.1)
        print(f"\n✓ Rebalancing check:")
        for greek_name, status in rebal.items():
            print(f"  - {greek_name}: {status['action']} (drift: {status['drift_pct']:.1f}%)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_kdb_client():
    """Test KDB+ fallback to CSV."""
    print("\n" + "="*60)
    print("Testing KDB+ Client (Fallback Mode)")
    print("="*60)
    
    try:
        from src.data.kdb_client import KDBClient, DataManager
        from pathlib import Path
        import tempfile
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Initialize client (will use CSV fallback)
            client = KDBClient(use_kdb=False, fallback_dir=tmpdir)
            print(f"✓ KDB Client initialized (fallback mode)")
            print(f"  - KDB available: {client.kdb_available}")
            print(f"  - Fallback dir: {client.fallback_dir}")
            
            # Create sample data
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            df = pd.DataFrame({
                'Open': np.random.uniform(100, 110, 100),
                'High': np.random.uniform(110, 120, 100),
                'Low': np.random.uniform(90, 100, 100),
                'Close': np.random.uniform(100, 110, 100),
                'Volume': np.random.uniform(1e6, 2e6, 100),
            }, index=dates)
            
            # Write data
            success = client.write_timeseries('TEST', df)
            print(f"\n✓ Data written to storage: {success}")
            
            # Read data back
            df_read = client.read_timeseries('TEST')
            print(f"✓ Data read from storage: {len(df_read)} rows")
            print(f"  - Columns: {df_read.columns.tolist()}")
            
            client.close()
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*60)
    print("█  RISK MANAGEMENT MODULE VALIDATION" + " "*12 + "█")
    print("█"*60)
    
    tests = [
        ("Greeks Calculator", test_greeks),
        ("Monte Carlo VaR/ES", test_monte_carlo_var),
        ("Risk Manager", test_risk_manager),
        ("KDB+ Client", test_kdb_client),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_indicator in results.items():
        status = "✓ PASS" if passed_indicator else "✗ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    print("█"*60)
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
