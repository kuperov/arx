from typing import List

import click
import jax
import pandas as pd

import arx.experiments as ex
from arx import cv


def full_bayes(
    experiment_no: int,
    experiment_variant: str,
    filename: str,
    T: int = 100,
    alternative: str = '10-fold',
    n_posts=10,
    mc_reps=500,
    n_warmup=400,
    n_chains=4,
    seed=0,
) -> None:
    """Compute and save individual draws for an experiment.

    Args:
        experiment_no:      The number of the experiment to run.
        experiment_variant: The variant of the experiment to run ("hard"/"easy").
        filename:           File to save results to
        T:                  Data length (number of simulated time periods).
        alternative:        Alternative model (either "10-fold" or "LOO")
        n_posts:            The number of individual datasets to simulate (ie number of posteriors).
        n_warmup:           Number of warmup iterations for MCMC
        n_chains:           Number of MCMC chains to run
        mc_reps:            Monte Carlo repetitions for estimating true elpd.
        seed:               The random seed to use.

    This experiment draws n samples from the true model, computes theoretical and CV
    estimates for the joint and pointwise elpds. Estimates are saved in FILENAME.

    For a given random seed and length, datasets are stable across invocations. This means
    different experiments can be guaranteed to use the same simulated datasets.
    """
    assert alternative in ["10-fold", "LOO"]
    model_key, data_key, sim_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    experiment = ex.make_full_experiment(experiment_no, experiment_variant)
    click.echo(
        f"Running experiment {experiment_no} {experiment_variant} with T={T} and seed={seed}"
    )
    draw_keys = jax.random.split(data_key, n_posts)
    sim_keys = jax.random.split(sim_key, n_posts)
    results = []

    def save():
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

    # Loop over y draws
    for i, draw_key in enumerate(draw_keys):
        for alpha in ex.ALPHA_SM:
            click.echo(f"Starting draw i={i} ({i+1}/{n_posts}) for alpha={alpha}")
            instance = experiment.make_full_instance(
                alpha=alpha, prng_key=model_key, model_params=dict(T=T)
            )
            y = instance.dgp.simulate(draw_key)
            mA_post = instance.mA.fd_nuts_post(
                y,
                prng_key=sim_keys[i],
                ndraws=mc_reps,
                nchains=n_chains,
                nwarmup=n_warmup,
            )
            mB_post = instance.mB.fd_nuts_post(
                y,
                prng_key=sim_keys[i],
                ndraws=mc_reps,
                nchains=n_chains,
                nwarmup=n_warmup,
            )
            mA = mA_post.estimate_elpd(instance.dgp, mA_post.arviz(), nreps=mc_reps)
            mB = mB_post.estimate_elpd(instance.dgp, mB_post.arviz(), nreps=mc_reps)
            tenfold = cv.KFoldCVScheme(T=instance.dgp.T, k=10)
            mA_tenfold = instance.mA.cv(tenfold, y)[0]
            mB_tenfold = instance.mB.cv(tenfold, y)[0]
            if alternative == "LOO":
                alt = cv.LOOCVScheme(T=instance.dgp.T)
            elif alternative == "10-fold":
                alt = cv.PointwiseKFoldCVScheme(T=instance.dgp.T, k=10)
            else:
                raise Exception(f"Unknown alternative scheme {alternative}")
            mA_loo = instance.mA.cv(alt, y)[0]
            mB_loo = instance.mB.cv(alt, y)[0]
            res = dict(
                index=i,
                alpha=alpha,
                mA_elppd=mA["mean_elppd"],
                mB_elppd=mB["mean_elppd"],
                mA_eljpd=mA["mean_eljpd"],
                mB_eljpd=mB["mean_eljpd"],
                mA_tenfold=mA_tenfold,
                mB_tenfold=mB_tenfold,
                mA_loo=mA_loo,
                mB_loo=mB_loo,
                T=T,
                n=n_posts,
                mc_reps=mc_reps,
                seed=seed,
            )
            # add model selection statistics
            for s in ["elppd", "eljpd", "tenfold", "loo"]:
                res[f"sel_{s}"] = res[f"mA_{s}"] - res[f"mB_{s}"]
            results.append(res)
            save()  # checkpoint
    click.echo(f"Saved results to {filename}")


def full_kfold(
    experiment_no: int,
    variant: str,
    filename: str,
    T: int = 100,
    k: int = 10,
    n_posts=10,
    mc_reps=500,
    mcmc_draws:int=1000,
    mcmc_chains:int=4,
    mcmc_warmup:int=500,
    seed=0,
) -> None:
    """Full Bayes k-fold experiment

    Args:
        experiment_no:      The number of the experiment to run.
        experiment_variant: The variant of the experiment to run ("hard"/"easy").
        filename:           File to save results to
        T:                  Data length (number of simulated time periods).
        k:                  Number of folds
        n_posts:            The number of individual datasets to simulate (ie number of posteriors).
        n_warmup:           Number of warmup iterations for MCMC
        n_chains:           Number of MCMC chains to run
        mc_reps:            Monte Carlo repetitions for estimating true elpd.
        seed:               The random seed to use.

    This experiment draws n samples from the true model, computes theoretical and CV
    estimates for the joint and pointwise elpds. Estimates are saved in FILENAME.

    For a given random seed and length, datasets are stable across invocations. This means
    different experiments can be guaranteed to use the same simulated datasets.
    """
    model_key, data_key, sim_key, sampling_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    experiment = ex.make_full_experiment(experiment_no, variant)
    click.echo(
        f"Running experiment {experiment_no} {variant} with T={T} and seed={seed}"
    )
    draw_keys = jax.random.split(data_key, n_posts)
    sim_keys = jax.random.split(sim_key, n_posts)
    results = []

    def save():
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

    # Loop over y draws
    for i, draw_key in enumerate(draw_keys):
        for alpha in ex.ALPHA_SM:
            click.echo(f"Starting draw i={i} ({i+1}/{n_posts}) for alpha={alpha}")
            instance = experiment.make_full_instance(
                alpha=alpha, prng_key=model_key, model_params=dict(T=T)
            )
            y = instance.dgp.simulate(draw_key)
            mA_post = instance.mA.fd_nuts_post(
                y,
                prng_key=sim_keys[i],
                ndraws=mcmc_draws,
                nchains=mcmc_chains,
                nwarmup=mcmc_warmup,
            )
            mB_post = instance.mB.fd_nuts_post(
                y,
                prng_key=sim_keys[i],
                ndraws=mcmc_draws,
                nchains=mcmc_chains,
                nwarmup=mcmc_warmup,
            )
            mA = mA_post.estimate_elpd(instance.dgp, mA_post.arviz(), nreps=mc_reps)
            mB = mB_post.estimate_elpd(instance.dgp, mB_post.arviz(), nreps=mc_reps)
            kfold = cv.KFoldCVScheme(T=instance.dgp.T, k=k)
            mA_joint = instance.mA.cv(kfold, y)[0]
            mB_joint = instance.mB.cv(kfold, y)[0]
            alt = cv.PointwiseKFoldCVScheme(T=instance.dgp.T, k=k)
            mA_pw = instance.mA.cv(alt, y)[0]
            mB_pw = instance.mB.cv(alt, y)[0]
            # # now do with mcmc as cross-check
            # FIXME: removing MCMC results for now because they don't make sense and need to be debugged
            # mA_mcmc = instance.mA.cv_nuts(scheme=kfold, y=y, prng_key=sampling_key, ndraws=mc_reps, nchains=n_chains, nwarmup=n_warmup)
            # mB_mcmc = instance.mB.cv_nuts(scheme=kfold, y=y, prng_key=sampling_key, ndraws=mc_reps, nchains=n_chains, nwarmup=n_warmup)
            res = dict(
                index=i,
                alpha=alpha,
                mA_elppd=mA["mean_elppd"],
                mB_elppd=mB["mean_elppd"],
                mA_eljpd=mA["mean_eljpd"],
                mB_eljpd=mB["mean_eljpd"],
                sel_elppd=mA["mean_elppd"] - mB["mean_elppd"],
                sel_eljpd=mA["mean_eljpd"] - mB["mean_eljpd"],
                mA_joint=mA_joint,
                mB_joint=mB_joint,
                sel_joint=mA_joint - mB_joint,
                mA_pw=mA_pw,
                mB_pw=mB_pw,
                sel_pw=mA_pw - mB_pw,
                # mA_mcmc_joint=mA_mcmc['joint'].elpdhat,
                # mB_mcmc_joint=mB_mcmc['joint'].elpdhat,
                # sel_joint_mcmc=mA_mcmc['joint'].elpdhat - mB_mcmc['joint'].elpdhat,
                # mA_mcmc_pw=mA_mcmc['pw'].elpdhat,
                # mB_mcmc_pw=mB_mcmc['pw'].elpdhat,
                # sel_pw_mcmc=mA_mcmc['pw'].elpdhat - mB_mcmc['pw'].elpdhat,
                T=T,
                n=n_posts,
                mc_reps=mc_reps,
                k=k,
                seed=seed,
            )
            results.append(res)
            save()  # checkpoint
    click.echo(f"Saved results to {filename}")
