# CV for autoregressions

Analysis of CV methods under time-series dependence.

You will need an environment that can run python 3.10 and JAX. The script
setup.sh might help with dependencies.

The script `setup.py` provides a shell command `arx` which is the entry point
for all experiments.

Once you've created a virtual environment and run `setup.py`, use `arx --help`
for usage information.

```
Usage: arx [OPTIONS] COMMAND [ARGS]...

  Run CV experiments

  This script is the entry point for all experiments in the paper. You'll
  probably need to run setup.sh (or similar) to make it work.

Options:
  --help  Show this message and exit.

Commands:
  by-alpha              Simplified model selection by alpha.
  by-dimension          Simplified model selection by dimension and alpha.
  by-excluded-effect    Simplified model selection by excluded effect.
  by-halo               Simplified model selection by halo.
  by-included-effect    Simplified model selection by included effect.
  by-length             Simplified model selection by data length.
  full-bayes            Compute draws for full-Bayes experiment.
  full-combine          Combine full-Bayes result files.
  full-kfold            Compute draws for k-fold full-Bayes experiment.
  generate-z            Generate Z for use across experiments.
  length-search         Simplified model selection by data length.
  loss                  Loss for simplified model selection by alpha.
  pointwise-comparison  Joint/pointwise loss comparison by alpha.
  supplementary         Joint/pointwise loss comparison by alpha.
```
