"""Guardrails module for the multi-agent stock analysis system.

Provides validation functions for workflow pre-flight checks, data completeness,
report quality, and cross-report consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from stock_analysis.schemas.analyst_reports import (
    AnalystReport,
    AnalystStance,
    AnalystType,
    DataGap,
    DataGapSeverity,
    DecisionType,
    MarketRegime,
    RiskCategory,
)
from stock_analysis.schemas.graph_state import GraphState


class GuardrailViolationType(str, Enum):
    """Types of guardrail violations."""
    MISSING_REPORTS = "missing_reports"
    CRITICAL_DATA_GAPS = "critical_data_gaps"
    INSUFFICIENT_DATA_COVERAGE = "insufficient_data_coverage"
    STANCE_CONFIDENCE_INCONSISTENCY = "stance_confidence_inconsistency"
    EXTREME_STANCE_IMBALANCE = "extreme_stance_imbalance"
    REPORT_QUALITY_FAILURE = "report_quality_failure"
    LOW_CONSENSUS_CONFIDENCE = "low_consensus_confidence"


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation."""
    violation_type: GuardrailViolationType
    message: str
    severity: str = "error"  # "error" or "warning"
    details: Optional[dict] = None


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    passed: bool
    violations: List[GuardrailViolation]
    warnings: List[GuardrailViolation]

    def __bool__(self) -> bool:
        return self.passed

    def add_violation(self, violation: GuardrailViolation) -> None:
        self.violations.append(violation)
        self.passed = False

    def add_warning(self, warning: GuardrailViolation) -> None:
        self.warnings.append(warning)


# ---------------------------------------------------------------------------
# Pre-flight guardrails
# ---------------------------------------------------------------------------

def validate_all_reports_present(state: GraphState) -> GuardrailResult:
    """Validate all 4 analyst reports are present."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])
    required_reports = {
        "technical_report": AnalystType.TECHNICAL,
        "fundamental_report": AnalystType.FUNDAMENTAL,
        "sentiment_report": AnalystType.SENTIMENT,
        "context_report": AnalystType.CONTEXT,
    }
    missing = []
    for field, analyst_type in required_reports.items():
        report = getattr(state, field, None)
        if report is None:
            missing.append(analyst_type.value)
    if missing:
        result.add_violation(GuardrailViolation(
            violation_type=GuardrailViolationType.MISSING_REPORTS,
            message=f"Missing analyst reports: {', '.join(missing)}",
            severity="error",
            details={"missing_reports": missing},
        ))
    return result


def validate_no_critical_data_gaps(state: GraphState) -> GuardrailResult:
    """Validate no unaddressed critical data gaps exist across all reports."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])
    reports = [
        state.technical_report,
        state.fundamental_report,
        state.sentiment_report,
        state.context_report,
    ]
    critical_gaps = []
    for report in reports:
        if report is None:
            continue
        try:
            analyst_report = AnalystReport(**report)
            for gap in analyst_report.data_gaps:
                if isinstance(gap, dict):
                    severity = gap.get("severity", "medium")
                elif isinstance(gap, DataGap):
                    severity = gap.severity.value
                else:
                    severity = "medium"
                if severity == DataGapSeverity.CRITICAL.value:
                    critical_gaps.append({
                        "analyst": analyst_report.analyst.value,
                        "description": gap.get("description") if isinstance(gap, dict) else gap.description,
                    })
        except Exception:
            continue
    if critical_gaps:
        result.add_violation(GuardrailViolation(
            violation_type=GuardrailViolationType.CRITICAL_DATA_GAPS,
            message=f"Critical data gaps detected: {len(critical_gaps)} gap(s)",
            severity="error",
            details={"critical_gaps": critical_gaps},
        ))
    return result


def validate_data_coverage(state: GraphState) -> GuardrailResult:
    """Validate minimum data coverage before allowing decision classification."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])
    reports = [
        state.technical_report,
        state.fundamental_report,
        state.sentiment_report,
        state.context_report,
    ]
    valid_reports = 0
    total_confidence = 0.0
    for report in reports:
        if report is None:
            continue
        try:
            analyst_report = AnalystReport(**report)
            if analyst_report.confidence > 0.3:
                valid_reports += 1
                total_confidence += analyst_report.confidence
        except Exception:
            continue
    if valid_reports < 2:
        result.add_violation(GuardrailViolation(
            violation_type=GuardrailViolationType.INSUFFICIENT_DATA_COVERAGE,
            message=f"Insufficient data coverage: only {valid_reports}/4 analysts have meaningful confidence (>0.3)",
            severity="error",
            details={"valid_reports": valid_reports, "total_reports": 4},
        ))
    elif valid_reports < 3:
        result.add_warning(GuardrailViolation(
            violation_type=GuardrailViolationType.INSUFFICIENT_DATA_COVERAGE,
            message=f"Limited data coverage: only {valid_reports}/4 analysts have meaningful confidence",
            severity="warning",
            details={"valid_reports": valid_reports, "total_reports": 4},
        ))
    return result


def validate_stance_distribution(state: GraphState) -> GuardrailResult:
    """Validate stance distribution is reasonable (not all extreme without neutral balance)."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])
    reports = [
        state.technical_report,
        state.fundamental_report,
        state.sentiment_report,
        state.context_report,
    ]
    stances = []
    extreme_count = 0
    for report in reports:
        if report is None:
            continue
        try:
            analyst_report = AnalystReport(**report)
            stances.append(analyst_report.stance)
            if analyst_report.stance in (AnalystStance.STRONG_BULLISH, AnalystStance.STRONG_BEARISH):
                extreme_count += 1
        except Exception:
            continue
    if not stances:
        return result
    if extreme_count == len(stances) and len(stances) >= 3:
        result.add_warning(GuardrailViolation(
            violation_type=GuardrailViolationType.EXTREME_STANCE_IMBALANCE,
            message=f"All {len(stances)} analysts have extreme stances - potential data quality issue",
            severity="warning",
            details={"stances": [s.value for s in stances], "extreme_count": extreme_count},
        ))
    return result


def validate_report_quality(state: GraphState) -> GuardrailResult:
    """Validate report quality (non-empty key_points, evidence, specific risks)."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])
    reports = [
        state.technical_report,
        state.fundamental_report,
        state.sentiment_report,
        state.context_report,
    ]
    for report in reports:
        if report is None:
            continue
        try:
            analyst_report = AnalystReport(**report)
            if not analyst_report.key_points:
                result.add_warning(GuardrailViolation(
                    violation_type=GuardrailViolationType.REPORT_QUALITY_FAILURE,
                    message=f"{analyst_report.analyst.value} report has no key points",
                    severity="warning",
                    details={"analyst": analyst_report.analyst.value},
                ))
            if not analyst_report.evidence:
                result.add_warning(GuardrailViolation(
                    violation_type=GuardrailViolationType.REPORT_QUALITY_FAILURE,
                    message=f"{analyst_report.analyst.value} report has no evidence",
                    severity="warning",
                    details={"analyst": analyst_report.analyst.value},
                ))
            if not analyst_report.risks:
                result.add_warning(GuardrailViolation(
                    violation_type=GuardrailViolationType.REPORT_QUALITY_FAILURE,
                    message=f"{analyst_report.analyst.value} report has no identified risks",
                    severity="warning",
                    details={"analyst": analyst_report.analyst.value},
                ))
        except Exception as e:
            result.add_violation(GuardrailViolation(
                violation_type=GuardrailViolationType.REPORT_QUALITY_FAILURE,
                message=f"Report validation failed: {e}",
                severity="error",
                details={"error": str(e)},
            ))
    return result


def validate_consensus_confidence(state: GraphState) -> GuardrailResult:
    """Validate that consensus confidence is sufficient for decision making."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])
    reports = [
        state.technical_report,
        state.fundamental_report,
        state.sentiment_report,
        state.context_report,
    ]
    confidences = []
    for report in reports:
        if report is None:
            continue
        try:
            analyst_report = AnalystReport(**report)
            confidences.append(analyst_report.confidence)
        except Exception:
            continue
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < 0.4:
            result.add_violation(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_CONSENSUS_CONFIDENCE,
                message=f"Low average analyst confidence: {avg_confidence:.2f}",
                severity="error",
                details={"avg_confidence": avg_confidence, "individual": confidences},
            ))
        elif avg_confidence < 0.5:
            result.add_warning(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_CONSENSUS_CONFIDENCE,
                message=f"Moderate average analyst confidence: {avg_confidence:.2f}",
                severity="warning",
                details={"avg_confidence": avg_confidence, "individual": confidences},
            ))
    return result


def run_preflight_guardrails(state: GraphState) -> GuardrailResult:
    """Run all pre-flight guardrails before decision engine."""
    result = GuardrailResult(passed=True, violations=[], warnings=[])

    # Run all guardrails
    checks = [
        validate_all_reports_present,
        validate_no_critical_data_gaps,
        validate_data_coverage,
        validate_stance_distribution,
        validate_report_quality,
        validate_consensus_confidence,
    ]

    for check in checks:
        check_result = check(state)
        result.violations.extend(check_result.violations)
        result.warnings.extend(check_result.warnings)
        if not check_result.passed:
            result.passed = False

    return result


# ---------------------------------------------------------------------------
# Data gap utilities
# ---------------------------------------------------------------------------

def normalize_data_gaps(gaps: List) -> List[DataGap]:
    """Normalize data gaps to DataGap objects."""
    normalized = []
    for gap in gaps:
        if isinstance(gap, DataGap):
            normalized.append(gap)
        elif isinstance(gap, dict):
            normalized.append(DataGap(**gap))
        elif isinstance(gap, str):
            # Try to infer severity from keywords
            severity = DataGapSeverity.MEDIUM
            lower_gap = gap.lower()
            if any(kw in lower_gap for kw in ["critical", "essential", "required", "missing price", "no data"]):
                severity = DataGapSeverity.CRITICAL
            elif any(kw in lower_gap for kw in ["limited", "incomplete", "partial"]):
                severity = DataGapSeverity.HIGH
            normalized.append(DataGap(description=gap, severity=severity))
    return normalized


def get_critical_gaps(reports: List[dict]) -> List[dict]:
    """Extract critical data gaps from analyst reports."""
    critical = []
    for report in reports:
        if report is None:
            continue
        try:
            analyst_report = AnalystReport(**report)
            for gap in analyst_report.data_gaps:
                severity = gap.severity if isinstance(gap, DataGap) else gap.get("severity", "medium")
                if severity == DataGapSeverity.CRITICAL:
                    critical.append({
                        "analyst": analyst_report.analyst.value,
                        "description": gap.description if isinstance(gap, DataGap) else gap.get("description", str(gap)),
                    })
        except Exception:
            continue
    return critical