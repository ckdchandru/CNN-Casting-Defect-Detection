from src.data.models import GateResult, ImbalanceReport

# Reviewed, not guessed.
MINIMUM_IMBALANCE_RATIO = 2.0
MINIMUM_MINORITY_SAMPLES = 200


class Phase1GateValidator:

    def validate(self, report: ImbalanceReport) -> GateResult:
        reasons = []

        if report.imbalance_ratio < MINIMUM_IMBALANCE_RATIO:
            reasons.append(
                f"ratio {report.imbalance_ratio:.2f} below "
                f"{MINIMUM_IMBALANCE_RATIO}"
            )

        if report.n_minority < MINIMUM_MINORITY_SAMPLES:
            reasons.append(
                f"minority count {report.n_minority} below "
                f"{MINIMUM_MINORITY_SAMPLES}"
            )

        if reasons:
            return GateResult(is_valid=False, message="; ".join(reasons))

        message = (
            f"ratio {report.imbalance_ratio:.2f}:1, "
            f"{report.n_minority} minority samples"
        )
        return GateResult(is_valid=True, message=message)
