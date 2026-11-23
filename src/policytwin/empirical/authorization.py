"""Opaque authorization for one verified empirical evaluation.

The serializable :class:`FrozenEvaluationInputs` object is evidence written to
an artifact, not an execution credential.  Real-data evaluation requires an
instance of :class:`FrozenEvaluationAuthorization` issued by the temporal
snapshot bridge while its no-network context is active.
"""

from __future__ import annotations

from typing import NoReturn

from policytwin.data.temporal_firewall import (
    AccessPurpose,
    FrozenSnapshot,
    TemporalFirewall,
)
from policytwin.empirical.schema import (
    EmpiricalEvaluationConfig,
    FrozenEvaluationInputs,
    FrozenInputError,
    FrozenPanel,
)

# Membership in this module-private identity registry is the minting proof.  It
# makes ``object.__new__(FrozenEvaluationAuthorization)`` insufficient to forge
# an execution credential.  Entries are removed before consumption checks so a
# failed or interrupted attempt cannot retry the same authority.
_ISSUED_AUTHORIZATIONS: set[FrozenEvaluationAuthorization] = set()


def _reject_construction() -> NoReturn:
    raise FrozenInputError(
        "real empirical authorization is opaque and may only be issued by "
        "the TemporalFirewall snapshot bridge"
    )


class FrozenEvaluationAuthorization:
    """Single-use, non-serializable authority for one real evaluation.

    The public constructor is deliberately unusable.  The bridge constructs an
    instance without invoking it only after it has verified the complete
    snapshot.  This is an API trust boundary, not a sandbox against arbitrary
    code executing inside the same Python interpreter.
    """

    __slots__ = (
        "_config_sha256",
        "_consumed",
        "_firewall",
        "_panel_sha256",
        "_receipt",
        "_snapshot",
    )

    _config_sha256: str
    _consumed: bool
    _firewall: TemporalFirewall
    _panel_sha256: str
    _receipt: FrozenEvaluationInputs
    _snapshot: FrozenSnapshot

    def __init__(self) -> None:
        _reject_construction()

    def __copy__(self) -> NoReturn:
        _reject_construction()

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        _reject_construction()

    def __reduce__(self) -> NoReturn:
        raise TypeError("FrozenEvaluationAuthorization cannot be serialized")

    @property
    def receipt(self) -> FrozenEvaluationInputs:
        """Return immutable evidence; this evidence is not itself authority."""
        return self._receipt

    def _consume(
        self,
        *,
        config: EmpiricalEvaluationConfig,
        panel: FrozenPanel,
    ) -> FrozenEvaluationInputs:
        """Consume the authority after re-verifying its bound snapshot."""
        if self not in _ISSUED_AUTHORIZATIONS:
            raise FrozenInputError(
                "frozen empirical authorization was not issued by the verified "
                "TemporalFirewall bridge or has already been consumed"
            )
        _ISSUED_AUTHORIZATIONS.remove(self)
        if self._consumed:
            raise FrozenInputError(
                "frozen empirical authorization has already been consumed"
            )
        if not self._firewall.frozen_evaluation_active:
            raise FrozenInputError(
                "real empirical evaluation requires the issuing no-network "
                "TemporalFirewall context to remain active"
            )
        if config.sha256 != self._config_sha256:
            raise FrozenInputError(
                "empirical config differs from the opaque authorization"
            )
        if panel.sha256 != self._panel_sha256:
            raise FrozenInputError(
                "empirical panel differs from the opaque authorization"
            )
        self._firewall.verify_snapshot(self._snapshot, AccessPurpose.FROZEN_EVALUATION)
        if self._snapshot.manifest_sha256 != self._receipt.data_manifest_sha256:
            raise FrozenInputError(
                "snapshot commitment changed after empirical authorization"
            )
        self._consumed = True
        return self._receipt


def _issue_frozen_evaluation_authorization(
    *,
    firewall: TemporalFirewall,
    snapshot: FrozenSnapshot,
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    receipt: FrozenEvaluationInputs,
) -> FrozenEvaluationAuthorization:
    """Issue one capability from the verified bridge; never export this helper."""
    if not firewall.frozen_evaluation_active:
        raise FrozenInputError(
            "authorization may only be issued inside frozen_evaluation()"
        )
    authorization = object.__new__(FrozenEvaluationAuthorization)
    authorization._config_sha256 = config.sha256
    authorization._consumed = False
    authorization._firewall = firewall
    authorization._panel_sha256 = panel.sha256
    authorization._receipt = receipt
    authorization._snapshot = snapshot
    _ISSUED_AUTHORIZATIONS.add(authorization)
    return authorization


__all__ = ["FrozenEvaluationAuthorization"]
