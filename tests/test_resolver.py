

from stock_analysis.market.resolver import ResolvedSymbol, SymbolResolver


class TestSymbolResolver:
    def setup_method(self):
        self.resolver = SymbolResolver()

    def test_resolve_exact_symbol_uppercase(self):
        result = self.resolver.resolve("RELIANCE")
        assert result is not None
        assert result.symbol == "RELIANCE"
        assert result.name == "Reliance Industries Ltd"
        assert result.exchange == "NSE"
        assert result.confidence == 1.0
        assert result.matched_by == "exact"

    def test_resolve_exact_symbol_lowercase(self):
        result = self.resolver.resolve("reliance")
        assert result is not None
        assert result.symbol == "RELIANCE"
        assert result.confidence == 1.0
        assert result.matched_by == "exact"

    def test_resolve_fuzzy_name(self):
        result = self.resolver.resolve("tata motors")
        assert result is not None
        assert result.symbol == "TATAMOTORS"
        assert result.name == "Tata Motors Ltd"
        assert result.confidence == 0.85
        assert result.matched_by == "fuzzy"

    def test_resolve_fuzzy_name_variations(self):
        result = self.resolver.resolve("Tata Motors")
        assert result is not None
        assert result.symbol == "TATAMOTORS"

    def test_resolve_invalid_symbol(self):
        result = self.resolver.resolve("INVALIDXYZ123")
        assert result is not None
        assert result.symbol == "INVALIDXYZ123.NS"
        assert result.confidence == 0.5
        assert result.matched_by == "fallback"

    def test_resolve_empty_string(self):
        result = self.resolver.resolve("")
        assert result is None

    def test_resolve_none(self):
        result = self.resolver.resolve(None)
        assert result is None

    def test_resolve_batch(self):
        inputs = ["RELIANCE", "tata motors", "INVALID", "TCS"]
        results = self.resolver.resolve_batch(inputs)
        assert len(results) == 4
        assert results["RELIANCE"].symbol == "RELIANCE"
        assert results["tata motors"].symbol == "TATAMOTORS"
        assert results["TCS"].symbol == "TCS"

    def test_is_valid_nse_symbol_known(self):
        assert self.resolver.is_valid_nse_symbol("RELIANCE") is True
        assert self.resolver.is_valid_nse_symbol("TCS") is True

    def test_is_valid_nse_symbol_unknown(self):
        assert self.resolver.is_valid_nse_symbol("UNKNOWN") is False

    def test_is_valid_nse_symbol_with_ns_suffix(self):
        assert self.resolver.is_valid_nse_symbol("RELIANCE.NS") is True

    def test_cache_works(self):
        result1 = self.resolver.resolve("RELIANCE")
        result2 = self.resolver.resolve("RELIANCE")
        assert result1 is result2

    def test_fuzzy_match_typos(self):
        result = self.resolver.resolve("relaince")
        assert result is not None
        assert result.symbol == "RELAINCE.NS"
        assert result.matched_by == "fallback"
        assert result.confidence == 0.5


class TestResolvedSymbol:
    def test_resolved_symbol_creation(self):
        symbol = ResolvedSymbol(
            symbol="RELIANCE",
            name="Reliance Industries Ltd",
            exchange="NSE",
            confidence=1.0,
            matched_by="exact"
        )
        assert symbol.symbol == "RELIANCE"
        assert symbol.confidence == 1.0
