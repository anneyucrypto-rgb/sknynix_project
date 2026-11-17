# SKHY–Korean Stock Cointegration Research Platform

This project builds point-in-time datasets for testing long-run equilibrium,
error correction, and directional price discovery between SKHY and SK hynix's
Korean ordinary shares (000660).

## Research datasets

| Dataset | Grain | Main consumer |
| --- | --- | --- |
| `daily` | Paired U.S. trading session | H1, H3 |
| `us_5min` | Completed SKHY five-minute bar | anchor selection, H2 |
| `anchors` | U.S. date × anchor type | H2, H3 |
| `h1` | Eligible daily observation | ARDL–UECM |
| `h2` | Date × anchor × horizon | conditional ECM |
| `h3` | Directional target observation | directional ECM |

All timestamps are stored in UTC. Session dates retain their market meaning,
and as-of joins only use observations available at the modeled information
cutoff. Korean ETF and currency series are explicitly labelled as market
proxies; they are not represented as direct Bank of Korea macro data.

## Build features

Install the dependencies in `requirements.txt`, then use the builder from the
project root:

```python
from src.engine.dataset_builder import DatasetBuilder
from src.engine.experiment import ExperimentSpec

spec = ExperimentSpec(
    hypothesis="H2",
    estimator="conditional_ecm",
    horizons=(5, 10, 15, 30, 45, 60, 100, 120),
)

with DatasetBuilder() as builder:
    result = builder.build(("h1", "h2", "h3"), experiment=spec)
```

Reusable outputs are written under `data/features/<dataset>/`. Experiment
datasets are isolated under
`data/features/experiments/run_id=<spec-fingerprint>/`; pass an explicit
`run_id` to `build()` to associate them with a model-registry run.
Build status is printed as each dependency starts, completes, or is restored
from cache. The cache key incorporates the feature version, experiment
specification, and immutable raw-snapshot content.

Dynamic anchors search completed five-minute bars ending from 07:05 through
09:20 U.S. Eastern and require consecutive liquidity-qualified intervals. The
fixed comparison anchor is the 09:30 U.S. Eastern opening bar. H2 accepts any
positive minute horizons, not only the defaults.

See [data/README.md](data/README.md) for the storage layout and
[data/MISSING_DATA.md](data/MISSING_DATA.md) for current source coverage.
