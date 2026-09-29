from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_analysis.schemas import (
    Symbol,
    SymbolCreate,
    SymbolUpdate,
    ForecastBaseline,
    ForecastLLM,
    ForecastFinal,
    PricePoint,
    ForecastCreate,
    ForecastUpdate,
    ForecastEvaluationCreate,
    PostmortemCreate,
    AuditLogCreate,
)


class TestSymbolSchemas:
    def test_symbol_create_valid(self):
        symbol = SymbolCreate(
            symbol="RELIANCE",
            name="Reliance Industries Ltd",
            sector="Energy",
            industry="Oil & Gas",
        )
        assert symbol.symbol == "RELIANCE"
        assert symbol.name == "Reliance Industries Ltd"
        assert symbol.sector == "Energy"
        assert symbol.exchange == "NSE"
        assert symbol.is_active is True

    def test_symbol_create_minimal(self):
        symbol = SymbolCreate(symbol="TCS", name="Tata Consultancy Services")
        assert symbol.symbol == "TCS"
        assert symbol.name == "Tata Consultancy Services"
        assert symbol.sector is None

    def test_symbol_create_invalid_symbol(self):
        with pytest.raises(ValidationError):
            SymbolCreate(symbol="invalid symbol!", name="Test")

    def test_symbol_create_empty_name(self):
        with pytest.raises(ValidationError):
            SymbolCreate(symbol="TEST", name="")

    def test_symbol_update_partial(self):
        update = SymbolUpdate(sector="Technology", is_active=False)
        assert update.sector == "Technology"
        assert update.is_active is False
        assert update.name is None

    def test_symbol_from_attributes(self):
        symbol = Symbol(
            id=1,
            symbol="INFY",
            name="Infosys Ltd",
            sector="Technology",
            industry="Software",
            exchange="NSE",
            is_active=True,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert symbol.id == 1
        assert symbol.symbol == "INFY"


class TestPricePoint:
    def test_price_point_full(self):
        pp = PricePoint(
            date=date(2024, 1, 15),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=1000000,
        )
        assert pp.close == Decimal("103.00")
        assert pp.volume == 1000000

    def test_price_point_minimal(self):
        pp = PricePoint(date=date(2024, 1, 15), close=Decimal("103.00"))
        assert pp.close == Decimal("103.00")
        assert pp.open is None
        assert pp.volume is None


class TestForecastBaseline:
    def test_forecast_baseline_valid(self):
        baseline = ForecastBaseline(
            symbol="RELIANCE",
            forecast_date=date(2024, 1, 15),
            horizon_days=5,
            predictions=[
                PricePoint(date=date(2024, 1, 16), close=Decimal("1000.00")),
                PricePoint(date=date(2024, 1, 17), close=Decimal("1010.00")),
            ],
            method="ARIMA",
            metadata={"order": (1, 1, 1)},
        )
        assert baseline.symbol == "RELIANCE"
        assert baseline.method == "ARIMA"
        assert len(baseline.predictions) == 2


class TestForecastLLM:
    def test_forecast_llm_valid(self):
        llm = ForecastLLM(
            symbol="RELIANCE",
            forecast_date=date(2024, 1, 15),
            horizon_days=5,
            predictions=[
                PricePoint(date=date(2024, 1, 16), close=Decimal("1005.00")),
            ],
            reasoning="Strong fundamentals support upside",
            confidence=0.75,
            adjustments={"news_impact": 0.02},
        )
        assert llm.confidence == 0.75
        assert llm.reasoning == "Strong fundamentals support upside"

    def test_forecast_llm_invalid_confidence(self):
        with pytest.raises(ValidationError):
            ForecastLLM(
                symbol="RELIANCE",
                forecast_date=date(2024, 1, 15),
                horizon_days=5,
                predictions=[],
                reasoning="test",
                confidence=1.5,
            )

    def test_forecast_llm_invalid_confidence_negative(self):
        with pytest.raises(ValidationError):
            ForecastLLM(
                symbol="RELIANCE",
                forecast_date=date(2024, 1, 15),
                horizon_days=5,
                predictions=[],
                reasoning="test",
                confidence=-0.1,
            )


class TestForecastFinal:
    def test_forecast_final_valid(self):
        final = ForecastFinal(
            symbol="RELIANCE",
            forecast_date=date(2024, 1, 15),
            horizon_days=5,
            predictions=[
                PricePoint(date=date(2024, 1, 16), close=Decimal("1002.00")),
            ],
            baseline_weight=0.6,
            llm_weight=0.4,
            confidence_score=0.7,
        )
        assert final.baseline_weight == 0.6
        assert final.llm_weight == 0.4
        assert final.confidence_score == 0.7

    def test_forecast_final_weight_sum(self):
        with pytest.raises(ValidationError):
            ForecastFinal(
                symbol="RELIANCE",
                forecast_date=date(2024, 1, 15),
                horizon_days=5,
                predictions=[],
                baseline_weight=0.8,
                llm_weight=0.5,
                confidence_score=0.7,
            )


class TestForecastSchemas:
    def test_forecast_create_valid(self):
        fc = ForecastCreate(
            symbol_id=1,
            forecast_date=date(2024, 1, 15),
            horizon_days=5,
            baseline_forecast_json='{"method": "ARIMA"}',
            final_forecast_json='{"predictions": []}',
        )
        assert fc.symbol_id == 1
        assert fc.status == "pending"

    def test_forecast_update_partial(self):
        fu = ForecastUpdate(
            llm_forecast_json='{"reasoning": "test"}',
            confidence_score=0.8,
        )
        assert fu.llm_forecast_json == '{"reasoning": "test"}'
        assert fu.confidence_score == 0.8
        assert fu.final_forecast_json is None


class TestEvaluationSchemas:
    def test_evaluation_create_valid(self):
        eval = ForecastEvaluationCreate(
            forecast_id=1,
            actual_prices_json='[{"date": "2024-01-16", "close": 1000}]',
            baseline_metrics_json='{"mae": 10.5}',
            final_metrics_json='{"mae": 8.2}',
            evaluation_date=date(2024, 1, 22),
        )
        assert eval.forecast_id == 1
        assert eval.llm_metrics_json is None


class TestPostmortemSchemas:
    def test_postmortem_create_valid(self):
        pm = PostmortemCreate(
            forecast_id=1,
            analysis_json='{"error": "missed earnings"}',
            lessons_learned="Watch earnings dates",
            calibration_adjustment_json='{"weight_shift": 0.1}',
        )
        assert pm.lessons_learned == "Watch earnings dates"


class TestAuditLogSchemas:
    def test_audit_log_create_valid(self):
        al = AuditLogCreate(
            entity_type="forecast",
            entity_id=1,
            action="created",
            changes_json='{"status": "pending"}',
            user_id="system",
        )
        assert al.entity_type == "forecast"
        assert al.action == "created"