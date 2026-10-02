import re
from dataclasses import dataclass
from typing import Optional

import yfinance as yf

from stock_analysis.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResolvedSymbol:
    symbol: str
    name: str
    exchange: str = "NSE"
    confidence: float = 1.0
    matched_by: str = "exact"


class SymbolResolver:
    COMMON_STOCKS = {
        "RELIANCE": "Reliance Industries Ltd",
        "TCS": "Tata Consultancy Services Ltd",
        "HDFCBANK": "HDFC Bank Ltd",
        "INFY": "Infosys Ltd",
        "ICICIBANK": "ICICI Bank Ltd",
        "HINDUNILVR": "Hindustan Unilever Ltd",
        "ITC": "ITC Ltd",
        "SBIN": "State Bank of India",
        "BHARTIARTL": "Bharti Airtel Ltd",
        "KOTAKBANK": "Kotak Mahindra Bank Ltd",
        "LT": "Larsen & Toubro Ltd",
        "ASIANPAINT": "Asian Paints Ltd",
        "AXISBANK": "Axis Bank Ltd",
        "MARUTI": "Maruti Suzuki India Ltd",
        "SUNPHARMA": "Sun Pharmaceutical Industries Ltd",
        "TITAN": "Titan Company Ltd",
        "ULTRACEMCO": "UltraTech Cement Ltd",
        "NESTLEIND": "Nestle India Ltd",
        "POWERGRID": "Power Grid Corporation of India Ltd",
        "NTPC": "NTPC Ltd",
        "TATAMOTORS": "Tata Motors Ltd",
        "TATASTEEL": "Tata Steel Ltd",
        "WIPRO": "Wipro Ltd",
        "BAJFINANCE": "Bajaj Finance Ltd",
        "HCLTECH": "HCL Technologies Ltd",
        "ADANIENT": "Adani Enterprises Ltd",
        "ADANIPORTS": "Adani Ports and Special Economic Zone Ltd",
        "COALINDIA": "Coal India Ltd",
        "ONGC": "Oil and Natural Gas Corporation Ltd",
        "JSWSTEEL": "JSW Steel Ltd",
        "TATACONSUM": "Tata Consumer Products Ltd",
        "TECHM": "Tech Mahindra Ltd",
        "BRITANNIA": "Britannia Industries Ltd",
        "CIPLA": "Cipla Ltd",
        "DIVISLAB": "Divi's Laboratories Ltd",
        "DRREDDY": "Dr. Reddy's Laboratories Ltd",
        "EICHERMOT": "Eicher Motors Ltd",
        "GRASIM": "Grasim Industries Ltd",
        "HEROMOTOCO": "Hero MotoCorp Ltd",
        "INDUSINDBK": "IndusInd Bank Ltd",
        "BAJAJFINSV": "Bajaj Finserv Ltd",
        "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd",
        "SBILIFE": "SBI Life Insurance Company Ltd",
        "HDFCLIFE": "HDFC Life Insurance Company Ltd",
        "TATAPOWER": "Tata Power Company Ltd",
        "BPCL": "Bharat Petroleum Corporation Ltd",
        "HINDALCO": "Hindalco Industries Ltd",
        "SHREECEM": "Shree Cement Ltd",
        "M&M": "Mahindra & Mahindra Ltd",
    }

    def __init__(self, enable_yfinance_validation: bool = True):
        self._symbol_cache: dict[str, ResolvedSymbol] = {}
        self._name_to_symbol: dict[str, str] = {}
        self._validation_cache: dict[str, bool] = {}
        self._enable_yfinance_validation = enable_yfinance_validation
        self._build_name_index()

    def _build_name_index(self) -> None:
        for symbol, name in self.COMMON_STOCKS.items():
            normalized_name = self._normalize_name(name)
            self._name_to_symbol[normalized_name] = symbol

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def _normalize_input(self, input_str: str) -> str:
        return re.sub(r"[^a-z0-9]", "", input_str.lower())

    def resolve(self, input_str: str) -> Optional[ResolvedSymbol]:
        if not input_str or not input_str.strip():
            return None

        input_str = input_str.strip().upper()

        if input_str in self._symbol_cache:
            return self._symbol_cache[input_str]

        normalized = self._normalize_input(input_str)

        if input_str in self.COMMON_STOCKS:
            result = ResolvedSymbol(
                symbol=input_str,
                name=self.COMMON_STOCKS[input_str],
                exchange="NSE",
                confidence=1.0,
                matched_by="exact"
            )
            self._symbol_cache[input_str] = result
            return result

        if normalized in self._name_to_symbol:
            symbol = self._name_to_symbol[normalized]
            result = ResolvedSymbol(
                symbol=symbol,
                name=self.COMMON_STOCKS[symbol],
                exchange="NSE",
                confidence=0.95,
                matched_by="name_exact"
            )
            self._symbol_cache[input_str] = result
            return result

        fuzzy_match = self._fuzzy_match(normalized)
        if fuzzy_match:
            result = ResolvedSymbol(
                symbol=fuzzy_match,
                name=self.COMMON_STOCKS[fuzzy_match],
                exchange="NSE",
                confidence=0.85,
                matched_by="fuzzy"
            )
            self._symbol_cache[input_str] = result
            return result

        yfinance_symbol = self._to_yfinance_symbol(input_str)
        if yfinance_symbol != input_str:
            if yfinance_symbol in self.COMMON_STOCKS:
                result = ResolvedSymbol(
                    symbol=yfinance_symbol,
                    name=self.COMMON_STOCKS[yfinance_symbol],
                    exchange="NSE",
                    confidence=0.8,
                    matched_by="yfinance_format"
                )
                self._symbol_cache[input_str] = result
                return result

        if self._enable_yfinance_validation:
            base_symbol = yfinance_symbol.replace(".NS", "").replace(".BO", "")
            if self._validate_with_yfinance(base_symbol):
                result = ResolvedSymbol(
                    symbol=yfinance_symbol,
                    name="",
                    exchange="NSE",
                    confidence=0.7,
                    matched_by="yfinance_validated"
                )
                self._symbol_cache[input_str] = result
                return result

        result = ResolvedSymbol(
            symbol=yfinance_symbol,
            name="",
            exchange="NSE",
            confidence=0.5,
            matched_by="fallback"
        )
        self._symbol_cache[input_str] = result
        return result

    def _validate_with_yfinance(self, symbol: str) -> bool:
        if symbol in self._validation_cache:
            return self._validation_cache[symbol]

        # Try with .NS suffix for NSE
        test_symbol = symbol if symbol.endswith(".NS") else symbol + ".NS"

        try:
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            is_valid = bool(info.get("symbol") or info.get("shortName") or info.get("longName"))
            self._validation_cache[symbol] = is_valid
            return is_valid
        except Exception as e:
            logger.debug("yfinance_validation_failed", symbol=symbol, error=str(e))
            self._validation_cache[symbol] = False
            return False

    def _fuzzy_match(self, normalized: str) -> Optional[str]:
        for name, symbol in self._name_to_symbol.items():
            if normalized in name or name in normalized:
                return symbol
        for name, symbol in self._name_to_symbol.items():
            if self._levenshtein_ratio(normalized, name) > 0.8:
                return symbol
        return None

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        max_len = max(len(s1), len(s2))
        return 1 - distances[-1] / max_len if max_len > 0 else 1.0

    def _to_yfinance_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().replace(" ", "")
        if symbol.endswith((".NS", ".BO")):
            return symbol
        return symbol.replace("-", "").replace(".", "") + ".NS"

    def resolve_batch(self, inputs: list[str]) -> dict[str, Optional[ResolvedSymbol]]:
        return {inp: self.resolve(inp) for inp in inputs}

    def is_valid_nse_symbol(self, symbol: str) -> bool:
        return symbol in self.COMMON_STOCKS or symbol.endswith(".NS")
