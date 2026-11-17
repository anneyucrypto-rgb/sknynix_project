from src.engine.dependency_resolver import resolve_dependencies
from src.engine.experiment import ExperimentSpec


def test_h3_dependency_order():
    order = resolve_dependencies("h3")
    assert order.index("daily") < order.index("h2") < order.index("h3")
    assert order.index("us_5min") < order.index("anchors") < order.index("h2")


def test_experiment_fingerprint_is_stable():
    spec = ExperimentSpec("H2", "conditional_ecm")
    assert spec.fingerprint == spec.fingerprint
    assert len(spec.fingerprint) == 16
