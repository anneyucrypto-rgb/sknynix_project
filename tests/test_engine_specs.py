from src.engine.dependency_resolver import resolve_dependencies
from src.engine.experiment import ExperimentSpec


def test_h3_dependency_order():
    order = resolve_dependencies("h3")
    assert order.index("daily") < order.index("h3")
    assert "h2" not in order
    assert "anchors" not in order


def test_experiment_fingerprint_is_stable():
    spec = ExperimentSpec("H2", "conditional_ecm")
    assert spec.fingerprint == spec.fingerprint
    assert len(spec.fingerprint) == 16


def test_fixed_h2_spec_accepts_full_sample_h1_coefficients():
    spec = ExperimentSpec(
        "H2", "fixed_30m_horizon_comparison",
        h1_model_id="full-sample-h1-v1", h1_intercept=1.0, h1_theta=0.8,
        horizons=(30,),
    )
    spec.validate()
    assert spec.horizons == (30,)
