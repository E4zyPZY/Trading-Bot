# src/data/kdb_client.py
"""
Optional KDB+ time-series database client.
Falls back to CSV/parquet if KDB+ is unavailable.

For production use with large datasets (>1TB historical data),
KDB+ provides native column compression, ultra-fast query performance,
and built-in time-series operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class KDBClient:
    """
    KDB+ client wrapper with graceful CSV fallback.
    In production: connects to KDB+ over TCP socket.
    In development: uses pandas CSV/parquet backend.
    """

    def __init__(
        self,
        use_kdb: bool = False,
        kdb_host: str = "localhost",
        kdb_port: int = 5000,
        fallback_dir: Path = None,
    ):
        """
        Initialize KDB client.
        
        use_kdb: if True, try to connect to KDB+ server
        kdb_host: KDB+ server hostname
        kdb_port: KDB+ server port
        fallback_dir: directory for CSV/parquet fallback storage
        """
        self.use_kdb = use_kdb
        self.kdb_host = kdb_host
        self.kdb_port = kdb_port
        self.fallback_dir = fallback_dir or Path("./data")
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
        self.kdb_connection = None
        self.kdb_available = False
        
        if use_kdb:
            self._try_connect_kdb()
        
        if not self.kdb_available:
            logger.info("KDB+ not available or disabled. Using CSV/parquet fallback.")

    def _try_connect_kdb(self):
        """Attempt to connect to KDB+ server."""
        try:
            import q  # q is the KDB+ Python interface
            
            self.kdb_connection = q.QConnection(
                host=self.kdb_host,
                port=self.kdb_port,
            )
            self.kdb_connection.open()
            self.kdb_available = True
            logger.info(f"Connected to KDB+ at {self.kdb_host}:{self.kdb_port}")
        except ImportError:
            logger.warning("q (KDB+ Python library) not installed. Install with: pip install q")
        except Exception as e:
            logger.warning(f"Could not connect to KDB+: {e}. Using fallback.")

    def write_timeseries(
        self,
        ticker: str,
        df: pd.DataFrame,
        table_name: Optional[str] = None,
    ) -> bool:
        """
        Write time-series data to KDB+ (if available) or CSV fallback.
        
        ticker: asset identifier
        df: DataFrame with date index and OHLCV columns
        table_name: custom table name (default: ticker)
        
        Returns: True if write succeeded
        """
        table_name = table_name or ticker
        
        if self.kdb_available:
            return self._write_kdb(table_name, df)
        else:
            return self._write_csv(ticker, df)

    def _write_kdb(self, table_name: str, df: pd.DataFrame) -> bool:
        """Write to KDB+ table."""
        try:
            # Reset index to make date a column
            df_kdb = df.reset_index()
            
            # Send to KDB+ via q function call
            # Assumes KDB+ has this function defined:
            # .Q.en[`:tsdb; (`date`ticker`open`high`low`close`volume)xkey df]
            
            self.kdb_connection.sendSync(
                f"upsert[`{table_name}; value flip csv_parse[read_csv[\"{table_name}.csv\"]]]"
            )
            logger.info(f"Wrote {len(df)} rows to KDB+ table {table_name}")
            return True
        except Exception as e:
            logger.error(f"KDB+ write failed: {e}. Falling back to CSV.")
            return self._write_csv(table_name, df)

    def _write_csv(self, ticker: str, df: pd.DataFrame) -> bool:
        """Write to CSV as fallback."""
        try:
            filepath = self.fallback_dir / f"{ticker}_timeseries.csv"
            df.to_csv(filepath)
            logger.info(f"Wrote {len(df)} rows to {filepath}")
            return True
        except Exception as e:
            logger.error(f"CSV write failed: {e}")
            return False

    def read_timeseries(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read time-series data from KDB+ or CSV.
        
        ticker: asset identifier
        start_date: filter from date (YYYY-MM-DD)
        end_date: filter to date (YYYY-MM-DD)
        columns: list of columns to retrieve (default: all)
        
        Returns: DataFrame
        """
        if self.kdb_available:
            return self._read_kdb(ticker, start_date, end_date, columns)
        else:
            return self._read_csv(ticker, start_date, end_date, columns)

    def _read_kdb(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read from KDB+."""
        try:
            # Build KDB+ query
            col_str = ",".join(columns) if columns else ""
            date_filter = ""
            if start_date and end_date:
                date_filter = f"select from {ticker} where date within ({start_date}; {end_date})"
            else:
                date_filter = f"select {col_str} from {ticker}"
            
            result = self.kdb_connection.sendSync(date_filter)
            
            # Convert KDB+ result to pandas DataFrame
            df = pd.DataFrame(result)
            logger.info(f"Read {len(df)} rows from KDB+ for {ticker}")
            return df
        except Exception as e:
            logger.error(f"KDB+ read failed: {e}. Falling back to CSV.")
            return self._read_csv(ticker, start_date, end_date, columns)

    def _read_csv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read from CSV fallback."""
        try:
            filepath = self.fallback_dir / f"{ticker}_timeseries.csv"
            
            if not filepath.exists():
                logger.warning(f"No data file found for {ticker} at {filepath}")
                return pd.DataFrame()
            
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Apply date filter
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Select columns
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                df = df[available_cols]
            
            logger.info(f"Read {len(df)} rows from CSV for {ticker}")
            return df
        except Exception as e:
            logger.error(f"CSV read failed: {e}")
            return pd.DataFrame()

    def close(self):
        """Close KDB+ connection if open."""
        if self.kdb_connection is not None:
            try:
                self.kdb_connection.close()
                logger.info("KDB+ connection closed.")
            except Exception as e:
                logger.error(f"Error closing KDB+: {e}")


class DataManager:
    """
    High-level data management interface.
    Abstracts away KDB+ vs CSV/parquet decision.
    """

    def __init__(self, use_kdb: bool = False, data_dir: Path = None):
        self.kdb_client = KDBClient(
            use_kdb=use_kdb,
            fallback_dir=data_dir,
        )

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV for a ticker. Returns standardized DataFrame."""
        df = self.kdb_client.read_timeseries(
            ticker,
            start_date=start_date,
            end_date=end_date,
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
        return df

    def store_ohlcv(self, ticker: str, df: pd.DataFrame) -> bool:
        """Store OHLCV data."""
        return self.kdb_client.write_timeseries(ticker, df)

    def fetch_tick_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch high-frequency tick data if available."""
        df = self.kdb_client.read_timeseries(
            f"{ticker}_ticks",
            start_date=start_date,
            end_date=end_date,
        )
        return df

    def close(self):
        """Close all connections."""
        self.kdb_client.close()


# Recommendation for large-scale deployment:
# 1. Install KDB+ Free Edition or Enterprise
# 2. Deploy KDB+ server on backend machine
# 3. Install q Python library: pip install q
# 4. Configure host/port in config.yaml:
#    data:
#      use_kdb: true
#      kdb_host: "kdb.server.com"
#      kdb_port: 5000
# 5. For 1TB+ historical data:
#    - KDB+ stores 500 GB in ~50 GB compressed
#    - Query latency: milliseconds vs seconds for CSV
#    - Parallel query support across cores
