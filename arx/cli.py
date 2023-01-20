#!.venv/bin/python3

from jax.config import config

config.update("jax_enable_x64", True)

import glob
import os
import pandas as pd

import click

import arx.arx_experiments as arxex
import arx.sarx_experiments as sarxex

RESULTS = 'results'

def ensure_results_dir():
    if not os.path.exists(RESULTS):
        os.mkdir(RESULTS)

@click.group()
def run_experiment() -> None:
    """Run CV experiments
    
    This script is the entry point for all experiments in the paper.
    You'll probably need to run setup.sh (or similar) to make it work.
    """

@click.command(name='generate-z')
@click.option('--max_len', default=5000, type=int, help='Number of rows')
@click.option('--max_q', default=5, type=int, help='Number of columns')
@click.option('--seed', default=0, type=int, help='Random seed (default 0)')
def generate_z(max_len: int, max_q, seed: int) -> None:
    """Generate Z for use across experiments."""
    sarxex.save_Z(max_len, max_q, seed)


@click.command(name="full-bayes")
@click.argument("experiment_no", type=int)
@click.argument("variant", type=click.Choice(['easy', 'hard']))
@click.argument("alternative", type=click.Choice(['LOO', '10-fold']))
@click.option("--t", type=int, default=100, help="Data length (default 100)")
@click.option("--n", type=int, default=10, help="Number of simulated posteriors (default 10)")
@click.option(
    "--mc-reps",
    type=int,
    default=1000,
    help="Number of MC reps for elpd (default 1000)",
)
@click.option("--seed", type=int, default=0, help="Random seed (default 0)")
def full(experiment_no: int, variant: str, alternative: str, t: int, n: int, mc_reps: int, seed: int) -> None:
    """Compute draws for full-Bayes experiment.

    This experiment draws n samples from the true model, computes theoretical and CV
    estimates for the joint and pointwise elpds. Estimates are saved as a file named
    results/{experiment_name}-t={t}-n={n}-mc={mc_reps}-seed={seed}.csv.

    For a given random seed and length, datasets are stable across invocations. This means
    different experiments can be guaranteed to use the same simulated datasets.

    To run this experiment, invoke this command ~50 times with different seeds in different
    threads. Then combine results using the full-combine command.
    """
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-alt={alternative}-T={t}-n={n}-mc={mc_reps}-seed={seed}.csv"
    arxex.full_bayes(experiment_no, variant, filename, t, alternative, n, mc_reps, seed)


@click.command(name="full-kfold")
@click.argument("experiment_no", type=int)
@click.argument("variant", type=click.Choice(['easy', 'hard']))
@click.option("--k", type=int, default=10, help="Number of folds (default 10)")
@click.option("--t", type=int, default=100, help="Data length (default 100)")
@click.option("--n", type=int, default=10, help="Number of simulated posteriors (default 10)")
@click.option("--mcmc_draws", type=int, default=1000, help="MCMC draws per chain (default 1000)")
@click.option("--mcmc_chains", type=int, default=4, help="MCMC chains (default 4)")
@click.option("--mcmc_warmup", type=int, default=1000, help="MCMC warmup (default 1000)")
@click.option(
    "--mc-reps",
    type=int,
    default=1000,
    help="Number of MC reps for elpd (default 1000)",
)
@click.option("--seed", type=int, default=0, help="Random seed (default 0)")
def full_kfold(experiment_no: int, variant: str, k: int, t: int, n: int, mc_reps: int, mcmc_draws:int, mcmc_chains:int, mcmc_warmup:int, seed: int) -> None:
    """Compute draws for k-fold full-Bayes experiment.

    This experiment draws n samples from the true model, computes theoretical and CV
    estimates for the joint and pointwise elpds. Estimates are saved as a file named
    results/{experiment_name}-t={t}-n={n}-{k}fold-mc={mc_reps}-seed={seed}.csv.

    For a given random seed and length, datasets are stable across invocations. This means
    different experiments can be guaranteed to use the same simulated datasets.

    To run this experiment, invoke this command ~50 times with different seeds in different
    threads. Then combine results using the full-combine command.
    """
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-{k}fold-T={t}-n={n}-mc={mc_reps}-seed={seed}.csv"
    arxex.full_kfold(
        experiment_no=experiment_no,
        variant=variant,
        filename=filename,
        T=t,
        k=k,
        n_posts=n,
        mc_reps=mc_reps,
        mcmc_draws=mcmc_draws,
        mcmc_chains=mcmc_chains,
        mcmc_warmup=mcmc_warmup,
        seed=seed)


@click.command(name="full-combine")
@click.argument("experiment_no", type=int)
@click.argument("variant", type=click.Choice(['easy', 'hard']))
def combine_individual(experiment_no: int, variant: str) -> None:
    """Combine full-Bayes result files."""
    outfile = f"{RESULTS}/{experiment_no}-{variant}-full-bayes.csv"
    infiles = glob.glob(f"{RESULTS}/{experiment_no}-{variant}-*-seed=*.csv")
    df = pd.concat([pd.read_csv(f) for f in infiles])
    df.to_csv(outfile, index=False)


@click.command(name='by-length')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def by_length(experiment_no: int, variant: str, seed: int = 0) -> None:
    """Simplified model selection by data length."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-by-length.csv"
    sarxex.by_data_length(filename=filename, ex_no=experiment_no, variant=variant, seed=seed)


@click.command(name='length-search')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
@click.option('--threshold', type=float, default=0.01, help='Threshold for length search (default 0.01)')
def length_search(experiment_no: int, variant: str, threshold: float, seed: int) -> None:
    """Simplified model selection by data length."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-length_search.csv"
    sarxex.length_search(filename=filename, ex_no=experiment_no, variant=variant, threshold_alpha=threshold, seed=seed)


@click.command(name='by-halo')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--lfo', is_flag=True, default=False, help='Use LFO (default False)')
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def by_halo(experiment_no: int, variant: str, lfo: bool, t: int, seed: int) -> None:
    """Simplified model selection by halo."""
    ensure_results_dir()
    suffix = '-lfo' if lfo else ''
    filename = f"{RESULTS}/{experiment_no}-{variant}-by-halo{suffix}.csv"
    sarxex.by_halo(
        filename=filename, ex_no=experiment_no, variant=variant, use_lfo=lfo, seed=seed, T=t)


@click.command(name='by-dimension')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--lfo', is_flag=True, default=False, help='Use LFO (default False)')
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def by_dimension(experiment_no: int, variant: str, lfo: bool, t: int, seed: int) -> None:
    """Simplified model selection by dimension and alpha."""
    ensure_results_dir()
    suffix = '-lfo' if lfo else ''
    filename = f"{RESULTS}/{experiment_no}-{variant}-by-dimension{suffix}.csv"
    sarxex.by_dimension(
        filename=filename, ex_no=experiment_no, variant=variant, use_lfo=lfo, T=t, seed=seed)


@click.command(name='by-alpha')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def by_alpha(experiment_no: int, variant: str, t: int, seed: int) -> None:
    """Simplified model selection by alpha."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-by-alpha.csv"
    sarxex.by_alpha(filename=filename, ex_no=experiment_no, variant=variant, T=t, seed=seed)


@click.command(name='by-included-effect')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--nreps', type=int, default=5_000, help='Number of repetitions (default 5_000)')
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def by_included_effect(experiment_no: int, variant: str, t: int, nreps: int, seed: int) -> None:
    """Simplified model selection by included effect."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-by-included-effect.csv"
    sarxex.by_included_effect(filename=filename, ex_no=experiment_no, variant=variant, T=t, seed=seed)


@click.command(name='by-excluded-effect')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--nreps', type=int, default=5_000, help='Number of repetitions (default 5_000)')
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def by_excluded_effect(experiment_no: int, variant: str, t: int, nreps: int, seed: int) -> None:
    """Simplified model selection by excluded effect."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-by-excluded-effect.csv"
    sarxex.by_excluded_effect(filename=filename, ex_no=experiment_no, variant=variant, T=t, seed=seed)


@click.command(name='loss')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--nreps', type=int, default=5_000, help='Number of repetitions (default 5_000)')
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def loss(experiment_no: int, variant: str, t: int, nreps: int, seed: int) -> None:
    """Loss for simplified model selection by alpha."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-loss.csv"
    sarxex.loss(filename=filename, ex_no=experiment_no, variant=variant, T=t, nreps=nreps, seed=seed)


@click.command(name='pointwise-comparison')
@click.argument('experiment_no', type=int)
@click.argument('variant', type=click.Choice(['easy', 'hard']))
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def pointwise_comparison(experiment_no: int, variant: str, t: int, seed: int) -> None:
    """Joint/pointwise loss comparison by alpha."""
    ensure_results_dir()
    filename = f"{RESULTS}/{experiment_no}-{variant}-pointwise-joint.csv"
    sarxex.pointwise_comparison(filename=filename, ex_no=experiment_no, variant=variant, t=t, seed=seed)


@click.command(name='supplementary')
@click.option('--t', type=int, default=100, help='Data length (default 100)')
@click.option("--seed", type=int, default=0, help='Random seed (default 0)')
def supplementary(t: int, seed: int) -> None:
    """Joint/pointwise loss comparison by alpha."""
    ensure_results_dir()
    filename = f"{RESULTS}/supplementary.csv"
    sarxex.supplementary(filename=filename, t=t, seed=seed)


run_experiment.add_command(generate_z)
run_experiment.add_command(full)
run_experiment.add_command(full_kfold)
run_experiment.add_command(combine_individual)
run_experiment.add_command(by_length)
run_experiment.add_command(by_halo)
run_experiment.add_command(by_dimension)
run_experiment.add_command(by_alpha)
run_experiment.add_command(loss)
run_experiment.add_command(by_included_effect)
run_experiment.add_command(by_excluded_effect)
run_experiment.add_command(length_search)
run_experiment.add_command(pointwise_comparison)
run_experiment.add_command(supplementary)


if __name__ == "__main__":
    run_experiment()
