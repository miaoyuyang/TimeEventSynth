"""Hard sanity checks for donor-policy dataset-pair construction."""

from __future__ import annotations

from typing import Any

from .donor_selection import is_cross_dataset_policy, is_same_dataset_only_policy
from .dataset_identity import pair_same_dataset


class DonorPolicySanityError(ValueError):
    """Raised when donor-policy pair construction violates experiment assumptions."""


def validate_donor_policy_pair_counts(
    donor_policy: str | None,
    *,
    same_dataset_pairs_considered: int,
    cross_dataset_pairs_considered: int,
) -> None:
    """Validate pair counts match donor_policy semantics."""
    policy = str(donor_policy or "")
    same = int(same_dataset_pairs_considered)
    cross = int(cross_dataset_pairs_considered)

    if is_cross_dataset_policy(policy):
        if cross <= 0:
            raise DonorPolicySanityError(
                "Cross-dataset policy did not evaluate any cross-dataset donor pairs. "
                "Check dataset metadata and donor-pool construction."
            )
        if same > 0:
            raise DonorPolicySanityError(
                f"Cross-dataset policy {policy!r} must not consider same-dataset pairs, "
                f"but same_dataset_pairs_considered={same}."
            )
        return

    if is_same_dataset_only_policy(policy):
        if same <= 0:
            raise DonorPolicySanityError(
                f"same_dataset_only policy did not evaluate any same-dataset donor pairs "
                f"(same={same}, cross={cross})."
            )
        if cross > 0:
            raise DonorPolicySanityError(
                f"same_dataset_only policy must not consider cross-dataset pairs, "
                f"but cross_dataset_pairs_considered={cross}."
            )


def validate_cross_dataset_accepted_rows(
    audit_rows: list[dict[str, Any]],
    *,
    donor_policy: str,
    feature_table: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Accepted synthesis rows under cross_dataset policies must use cross-dataset donors only."""
    if not is_cross_dataset_policy(donor_policy):
        return
    violations: list[str] = []
    for row in audit_rows:
        if row.get("record_type") == "donor_pair":
            if bool(row.get("same_dataset", False)):
                violations.append(str(row.get("candidate_id")))
            continue
        if not bool(row.get("accepted", row.get("kept", False))):
            continue
        if str(row.get("rejection_stage", "")) == "compatibility":
            continue
        target_id = str(row.get("target_series_id", ""))
        if feature_table:
            for donor_id in str(row.get("donor_series_ids", "") or "").split("|"):
                if donor_id and pair_same_dataset(target_id, donor_id, feature_table):
                    violations.append(f"{row.get('candidate_id')}:donor={donor_id}")
        elif bool(row.get("same_dataset", False)):
            violations.append(str(row.get("candidate_id", target_id)))
    if violations:
        raise DonorPolicySanityError(
            f"cross_dataset policy {donor_policy!r} accepted same-dataset donor usage in: "
            + ", ".join(violations[:10])
        )
