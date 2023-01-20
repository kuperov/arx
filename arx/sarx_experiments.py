import jax.numpy as jnp
import pandas as pd

from arx.cv import *
from arx.experiments import *
from arx.sarx import *


def by_excluded_effect(filename, ex_no: int, variant: str, T: int = 100, seed: int = 0):
    """Simplified model selection experiment, varying beta2 (the excluded effect)

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        T:        The length of the data
        seed:     The random seed. 
    """
    key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    ex = make_simplified_experiment(ex_no, variant)
    results = []
    for alpha in ALPHA_SM:
        for beta2 in EFFECT_SIZES:
            beta_star = jnp.array([1.0, 1.0, beta2])
            ei = ex.make_simplified_instance(
                alpha, prng_key=key, model_params=dict(beta_star=beta_star)
            )
            schemes = [
                LOOCVScheme(ei.dgp.T),
                HVBlockCVScheme(ei.dgp.T, h=3, v=3),
                HVBlockCVScheme(ei.dgp.T, h=3, v=0),
                KFoldCVScheme(ei.dgp.T, k=5),
                KFoldCVScheme(ei.dgp.T, k=10),
                LFOCVScheme(ei.dgp.T, h=0, v=0, w=10),
                LFOCVScheme(ei.dgp.T, h=3, v=3, w=10),
                ]
            for scheme in schemes:
                bm = ei.mA.eljpd_cv_benchmark(scheme) - ei.mB.eljpd_cv_benchmark(scheme)
                cv = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
                cv_qs, cv_ns = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                bm_qs, bm_ns = bm.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                err = cv - bm
                res = dict(
                    scheme=scheme.name(),
                    beta2=beta2,
                    alpha=alpha,
                    bmark_mean=bm.mean(),
                    bmark_std=bm.std(),
                    cv_mean=cv.mean(),
                    cv_std=cv.std(),
                    err_mean=err.mean(),
                    err_std=err.std(),
                    sigsq_star=ei.dgp.sigsq_star,
                    sigsq_mA=ei.mA.sigsq_hat,
                    sigsq_mB=ei.mB.sigsq_hat,
                    phi_mA=ei.mA.phi_hat,
                    phi_mB=ei.mB.phi_hat,
                    cv_lower_q=cv_qs[0],
                    cv_upper_q=cv_qs[-1],
                    cv_negshare=cv_ns,
                    bm_lower_q=bm_qs[0],
                    bm_upper_q=bm_qs[-1],
                    bm_negshare=bm_ns,
                )
                for i, ph in enumerate(ei.dgp.phi_star):
                    res[f"phi_star_{i+1}"] = float(ph)
                results.append(res)
                # cumulatively checkpoint results
                pd.DataFrame(results).to_csv(filename, index=False)
    return pd.DataFrame(results)


def by_included_effect(filename, ex_no: int, variant: str, T: int = 100, seed: int = 0):
    """Simplified model selection experiment, varying beta1 (the included effect)

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        T:        The length of the data
        seed:     The random seed. 
    """
    key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    ex = make_simplified_experiment(ex_no, variant)
    results = []
    for alpha in ALPHA_SM:
        for beta0 in EFFECT_SIZES:
            beta_star = jnp.array([beta0, 1.0, 1.0])
            ei = ex.make_simplified_instance(
                alpha, prng_key=key, model_params=dict(beta_star=beta_star)
            )
            schemes = [
                LOOCVScheme(ei.dgp.T),
                HVBlockCVScheme(ei.dgp.T, h=3, v=3),
                HVBlockCVScheme(ei.dgp.T, h=3, v=0),
                KFoldCVScheme(ei.dgp.T, k=5),
                KFoldCVScheme(ei.dgp.T, k=10),
                LFOCVScheme(ei.dgp.T, h=0, v=0, w=10),
                LFOCVScheme(ei.dgp.T, h=3, v=3, w=10),
                ]
            for scheme in schemes:
                bm = ei.mA.eljpd_cv_benchmark(scheme) - ei.mB.eljpd_cv_benchmark(scheme)
                cv = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
                err = cv - bm
                cv_qs, cv_ns = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                bm_qs, bm_ns = bm.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                res = dict(
                    scheme=scheme.name(),
                    alpha=alpha,
                    beta0=beta0,
                    bmark_mean=bm.mean(),
                    bmark_std=bm.std(),
                    cv_mean=cv.mean(),
                    cv_std=cv.std(),
                    err_mean=err.mean(),
                    err_std=err.std(),
                    sigsq_star=ei.dgp.sigsq_star,
                    sigsq_mA=ei.mA.sigsq_hat,
                    sigsq_mB=ei.mB.sigsq_hat,
                    phi_mA=ei.mA.phi_hat,
                    phi_mB=ei.mB.phi_hat,
                    cv_lower_q=cv_qs[0],
                    cv_upper_q=cv_qs[-1],
                    cv_negshare=cv_ns,
                    bm_lower_q=bm_qs[0],
                    bm_upper_q=bm_qs[-1],
                    bm_negshare=bm_ns,
                )
                for i, ph in enumerate(ei.dgp.phi_star):
                    res[f"phi_star_{i+1}"] = float(ph)
                results.append(res)
                # cumulatively checkpoint results
                pd.DataFrame(results).to_csv(filename, index=False)
    return pd.DataFrame(results)


def by_data_length(filename, ex_no: int, variant: str, seed: int = 0):
    """Simplified model selection experiment, varying time

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        seed:     The random seed. 
    """
    key = jax.random.PRNGKey(seed)
    ex = make_simplified_experiment(ex_no, variant)
    # support restarts by loading existing results
    existing_results = pd.read_csv(filename) if os.path.exists(filename) else None
    results = existing_results.to_dict('records') if existing_results is not None else []
    done = set(f"{r['scheme']}-{r['T']}-{r['alpha']:.2f}" for r in results)
    zkey, ekey = jax.random.split(key)
    # we do it this way to keep zkey the same for all experiments
    model_key, sim_key = jax.random.split(ekey)
    Zall = load_Z(q=ex.model_params['q'], T=max(DATA_LENGTHS))
    for alpha in ALPHA_SM:
        for t in DATA_LENGTHS:
            this_Z = Zall[:t,:]
            ei = ex.make_simplified_instance(
                alpha, prng_key=model_key, model_params=dict(T=t, Z=this_Z)
            )
            schemes = [
                LOOCVScheme(ei.dgp.T),
                HVBlockCVScheme(ei.dgp.T, h=3, v=3),
                # HVBlockCVScheme(ei.dgp.T, h=3, v=0),
                # KFoldCVScheme(ei.dgp.T, k=5),
                # KFoldCVScheme(ei.dgp.T, k=10),
                # LFOCVScheme(ei.dgp.T, h=0, v=0, m=10),
                # LFOCVScheme(ei.dgp.T, h=3, v=3, m=10),
                ]
            for scheme in schemes:
                if f"{scheme.name()}-{t}-{alpha:.2f}" in done:
                    print(f"Skipping {scheme.name()} for T={t}, alpha={alpha:.2f}")
                    continue
                else:
                    print(f"{scheme.name()} for T={t}, alpha={alpha:.2f}")
                bm = ei.mA.eljpd_cv_benchmark(scheme) - ei.mB.eljpd_cv_benchmark(scheme)
                cv = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
                err = cv - bm
                cv_qs, cv_ns = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                bm_qs, bm_ns = bm.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                res = dict(
                    T=t,
                    alpha=alpha,
                    scheme=scheme.name(),
                    bmark_mean=bm.mean(),
                    bmark_std=bm.std(),
                    cv_mean=cv.mean(),
                    cv_std=cv.std(),
                    err_mean=err.mean(),
                    err_std=err.std(),
                    sigsq_star=ei.dgp.sigsq_star,
                    sigsq_mA=ei.mA.sigsq_hat,
                    sigsq_mB=ei.mB.sigsq_hat,
                    phi_mA=ei.mA.phi_hat,
                    phi_mB=ei.mB.phi_hat,
                    cv_lower_q=cv_qs[0],
                    cv_upper_q=cv_qs[-1],
                    cv_negshare=cv_ns,
                    bm_lower_q=bm_qs[0],
                    bm_upper_q=bm_qs[-1],
                    bm_negshare=bm_ns,
                )
                for i, ph in enumerate(ei.dgp.phi_star):
                    res[f"phi_star_{i+1}"] = float(ph)
                results.append(res)
                done.add(f"{scheme.name()}-{t}-{alpha:.2f}")
                # cumulatively checkpoint results
                pd.DataFrame(results).to_csv(filename, index=False)
    return pd.DataFrame(results)


def length_search(filename: str, ex_no: int, variant: str, threshold_alpha=0.01, seed: int = 0):
    """Simplified model selection experiment, varying time

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        threshold_alpha: separatedness threshold (default 0.01)
        seed:     The random seed
    """
    key = jax.random.PRNGKey(seed)
    ex = make_simplified_experiment(ex_no, variant)
    existing_results = pd.read_csv(filename) if os.path.exists(filename) else None
    results = existing_results.to_dict('records') if existing_results is not None else []
    is_done = set(f'{r["scheme"]}-{r["alpha"]}' for r in results)
    zkey, ekey = jax.random.split(key)
    # we do it this way to keep zkey the same for all experiments
    model_key, sim_key = jax.random.split(ekey)
    Zall = load_Z(q=ex.model_params['q'], T=5000)
    scheme_factories = [  # ~~java vibes~~
        ('LOO', lambda t: LOOCVScheme(t)),
        ('hv-block', lambda t: HVBlockCVScheme(t, h=3, v=3)),
        ('h-block', lambda t: HVBlockCVScheme(t, h=3, v=0)),
        ('5-fold', lambda t: KFoldCVScheme(t, k=5)),
        ('10-fold', lambda t: KFoldCVScheme(t, k=10)),
        ('LFO-pw', lambda t: LFOCVScheme(t, h=0, v=0, w=10)),
        ('LFO-joint', lambda t: LFOCVScheme(t, h=3, v=3, w=10)),
    ]
    lbounds = {scheme_name: 4 for scheme_name, _ in scheme_factories}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for scheme_name, scheme_factory in scheme_factories:
            def is_well_separated_for_len(t):
                this_Z = Zall[:t,:]
                ei = ex.make_simplified_instance(
                    alpha, prng_key=model_key, model_params=dict(T=t, Z=this_Z)
                )
                scheme = scheme_factory(int(ei.dgp.T))
                cv = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
                _, cv_ns = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                is_sep = bool(cv_ns < threshold_alpha)  # manually unbox jax value
                del cv_ns, cv, scheme, ei, this_Z  # pray to memory leak deities
                return is_sep
            if f'{scheme_name}-{alpha}' in is_done:
                print(f"*** skipping search for alpha={alpha}, scheme={scheme_name}")
                existing_Tmin = existing_results[(existing_results.alpha==alpha) & (existing_results.scheme==scheme_name)]['Tmin'].min()
                lbounds[scheme_name] = max(lbounds[scheme_name], existing_Tmin)
                continue
            # binaryish search - first cutpoint is off-center at 300 to speed up better cases
            # and search window starts close to previous minimum
            L, R, m = max(4, lbounds[scheme_name]-10), 2400, max(300, lbounds[scheme_name] + 100)
            print(f"*** starting search for alpha={alpha}, scheme={scheme_name}, {L}-{R}")
            while L < R:
                print(f"    L={L}, R={R}, m={m}")
                if is_well_separated_for_len(m):
                    R = m
                else:
                    L = m + 1
                m = int((L + R) // 2)
            print(f"alpha={alpha}, scheme={scheme_name}, Tmin = {m}")
            lbounds[scheme_name] = m
            results.append(dict(
                alpha=float(alpha),
                scheme=scheme_name,
                Tmin=m,
                threshold_alpha=threshold_alpha,
                ex_no=ex_no,
                variant=variant,
                seed=seed,
            ))
            is_done.add(f'{scheme_name}-{alpha}')  # we don't actually read this back in
            pd.DataFrame(results).to_csv(filename, index=False)
            del is_well_separated_for_len 


def pointwise_comparison(filename: str, ex_no: int, variant: str, t=100, seed: int = 0):
    """Compare joint and pointwise performance of various schemes

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        t:        Data length
        variant:  Which variant (hard/easy)
        seed:     The random seed
    """
    key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    ex = make_simplified_experiment(ex_no, variant)
    results = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ei = ex.make_simplified_instance(
            alpha, prng_key=key, model_params=dict(T=t)
        )
        schemes = [
            HVBlockCVScheme(ei.dgp.T, h=3, v=3),
            KFoldCVScheme(ei.dgp.T, k=5),
            KFoldCVScheme(ei.dgp.T, k=10),
            LFOCVScheme(ei.dgp.T, h=3, v=3, w=10),
        ]
        for scheme in schemes:
            joint = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
            pw = ei.mA.eljpdhat_cv(scheme, pointwise=True) - ei.mB.eljpdhat_cv(scheme, pointwise=True)
            j_qs, j_ns = joint.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
            pw_qs, pw_ns = pw.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
            res = dict(
                scheme=scheme.name(),
                alpha=alpha,
                sigsq_star=ei.dgp.sigsq_star,
                sigsq_mA=ei.mA.sigsq_hat,
                sigsq_mB=ei.mB.sigsq_hat,
                phi_mA=ei.mA.phi_hat,
                phi_mB=ei.mB.phi_hat,
                j_lower_q=j_qs[0],
                j_upper_q=j_qs[-1],
                j_negshare=j_ns,
                pw_lower_q=pw_qs[0],
                pw_upper_q=pw_qs[-1],
                pw_negshare=pw_ns,
            )
            results.append(res)
            pd.DataFrame(results).to_csv(filename, index=False)


def by_halo(filename, ex_no: int, variant: str, use_lfo: bool = False, T: int = 100, seed: int = 0):
    """Simplified model selection experiment, varying h (halo)

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        use_lfo:  Use LFO instead of hv-block
        T:        The length of the data
        seed:     The random seed. 
    """
    key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    ex = make_simplified_experiment(ex_no, variant)
    results = []
    for alpha in ALPHA_SM:
        for v in [0, 3]:
            for h in HALOS:
                ei = ex.make_simplified_instance(
                    alpha, prng_key=key, model_params=dict())
                if use_lfo:
                    scheme = LFOCVScheme(ei.dgp.T, h=h, v=v, w=10)
                else:
                    scheme = HVBlockCVScheme(ei.dgp.T, h=h, v=v)
                bm = ei.mA.eljpd_cv_benchmark(scheme) - ei.mB.eljpd_cv_benchmark(scheme)
                cv = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
                cv_qs, cv_ns = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                bm_qs, bm_ns = bm.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                err = cv - bm
                res = dict(
                    h=h,
                    v=v,
                    alpha=alpha,
                    scheme=scheme.name(),
                    bmark_mean=bm.mean(),
                    bmark_std=bm.std(),
                    cv_mean=cv.mean(),
                    cv_std=cv.std(),
                    err_mean=err.mean(),
                    err_std=err.std(),
                    sigsq_star=ei.dgp.sigsq_star,
                    sigsq_mA=ei.mA.sigsq_hat,
                    sigsq_mB=ei.mB.sigsq_hat,
                    phi_mA=ei.mA.phi_hat,
                    phi_mB=ei.mB.phi_hat,
                    cv_lower_q=cv_qs[0],
                    cv_upper_q=cv_qs[-1],
                    cv_negshare=cv_ns,
                    bm_lower_q=bm_qs[0],
                    bm_upper_q=bm_qs[-1],
                    bm_negshare=bm_ns,
                )
                for i, ph in enumerate(ei.dgp.phi_star):
                    res[f"phi_star_{i+1}"] = float(ph)
                results.append(res)
                # cumulatively checkpoint results
                pd.DataFrame(results).to_csv(filename, index=False)
    return pd.DataFrame(results)


def by_dimension(filename, ex_no: int, variant: str, use_lfo: bool = False, T: int = 100, seed: int = 0):
    """Simplified model selection experiment, varying h (halo)

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        use_lfo:  Use LFO instead of hv-block
        T:        The length of the data
        seed:     The random seed. 
    """
    key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    ex = make_simplified_experiment(ex_no, variant)
    results = []
    for alpha in ALPHA_SM:
        for h in [0, 3]:
            for v in DIMENSIONS:
                ei = ex.make_simplified_instance(
                    alpha, prng_key=key, model_params=dict())
                if use_lfo:
                    scheme = LFOCVScheme(ei.dgp.T, h=h, v=v, w=10)
                else:
                    scheme = HVBlockCVScheme(ei.dgp.T, h=h, v=v)
                bm = ei.mA.eljpd_cv_benchmark(scheme) - ei.mB.eljpd_cv_benchmark(scheme)
                cv = ei.mA.eljpdhat_cv(scheme) - ei.mB.eljpdhat_cv(scheme)
                cv_qs, cv_ns = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                bm_qs, bm_ns = bm.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                err = cv - bm
                res = dict(
                    v=v,
                    h=h,
                    alpha=alpha,
                    scheme=scheme.name(),
                    bmark_mean=bm.mean(),
                    bmark_std=bm.std(),
                    cv_mean=cv.mean(),
                    cv_std=cv.std(),
                    err_mean=err.mean(),
                    err_std=err.std(),
                    sigsq_star=ei.dgp.sigsq_star,
                    sigsq_mA=ei.mA.sigsq_hat,
                    sigsq_mB=ei.mB.sigsq_hat,
                    phi_mA=ei.mA.phi_hat,
                    phi_mB=ei.mB.phi_hat,
                    cv_lower_q=cv_qs[0],
                    cv_upper_q=cv_qs[-1],
                    cv_negshare=cv_ns,
                    bm_lower_q=bm_qs[0],
                    bm_upper_q=bm_qs[-1],
                    bm_negshare=bm_ns,
                )
                for i, ph in enumerate(ei.dgp.phi_star):
                    res[f"phi_star_{i+1}"] = float(ph)
                results.append(res)
                # cumulatively checkpoint results
                pd.DataFrame(results).to_csv(filename, index=False)
    return pd.DataFrame(results)


def by_alpha(filename, ex_no: int, variant: str, T: int = 100, seed: int = 0):
    """Simplified model selection experiment, varying alpha

    Args:
        filename: File to save results to (csv)
        ex_no:    The experiment number
        variant:  Which variant (hard/easy)
        T:        The length of the data
        seed:     The random seed. 
    """
    key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    ex = make_simplified_experiment(ex_no, variant)
    results = []
    def save():
        pd.DataFrame(results).to_csv(filename, index=False)
    cv_schemes = [
        LOOCVScheme(T),
        HVBlockCVScheme(T, h=3, v=3),
        HVBlockCVScheme(T, h=3, v=0),
        KFoldCVScheme(T, k=5),
        KFoldCVScheme(T, k=10),
        LFOCVScheme(T, h=0, v=0, w=10),
        LFOCVScheme(T, h=3, v=3, w=10),
    ]
    for alpha in ALPHA_RANGE:
        ei = ex.make_simplified_instance(
            alpha, prng_key=key, model_params=dict())
        # joint difference
        I_T = jnp.eye(ei.dgp.T)
        jd : Poly = ei.mA.eljpd(I_T) - ei.mB.eljpd(I_T)
        qs, negshare = jd.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
        res = dict(
            method='theoretical',
            scheme='eljpd',
            alpha=alpha,
            bmark_mean=jd.mean(),
            bmark_std=jd.std(),
            cv_mean=None,
            cv_std=None,
            err_mean=None,
            err_std=None,
            sigsq_star=ei.dgp.sigsq_star,
            sigsq_mA=ei.mA.sigsq_hat,
            sigsq_mB=ei.mB.sigsq_hat,
            phi_mA=ei.mA.phi_hat,
            phi_mB=ei.mB.phi_hat,
            lower_q=qs[0],
            upper_q=qs[-1],
            negshare=negshare,
        )
        results.append(res)
        # hack to construct elppd
        loo = LOOCVScheme(T)
        pwd = ei.mA.eljpd_cv_benchmark(loo) - ei.mB.eljpd_cv_benchmark(loo)
        qs, negshare = pwd.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
        res = dict(
            method='theoretical',
            scheme='elppd',
            alpha=alpha,
            bmark_mean=pwd.mean(),
            bmark_std=pwd.std(),
            cv_mean=None,
            cv_std=None,
            err_mean=None,
            err_std=None,
            sigsq_star=ei.dgp.sigsq_star,
            sigsq_mA=ei.mA.sigsq_hat,
            sigsq_mB=ei.mB.sigsq_hat,
            phi_mA=ei.mA.phi_hat,
            phi_mB=ei.mB.phi_hat,
            lower_q=qs[0],
            upper_q=qs[-1],
            negshare=negshare,
        )
        results.append(res)
        for sch in cv_schemes:
            bm = ei.mA.eljpd_cv_benchmark(sch) - ei.mB.eljpd_cv_benchmark(sch)
            cv = ei.mA.eljpdhat_cv(sch) - ei.mB.eljpdhat_cv(sch)
            err = cv - bm
            qs, negshare = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
            res = dict(
                method='cv',
                scheme=sch.name(),
                alpha=alpha,
                bmark_mean=bm.mean(),
                bmark_std=bm.std(),
                cv_mean=cv.mean(),
                cv_std=cv.std(),
                err_mean=err.mean(),
                err_std=err.std(),
                sigsq_star=ei.dgp.sigsq_star,
                sigsq_mA=ei.mA.sigsq_hat,
                sigsq_mB=ei.mB.sigsq_hat,
                phi_mA=ei.mA.phi_hat,
                phi_mB=ei.mB.phi_hat,
                lower_q=qs[0],
                upper_q=qs[-1],
                negshare=negshare,
            )
            for i, ph in enumerate(ei.dgp.phi_star):
                res[f"phi_star_{i+1}"] = float(ph)
            results.append(res)
            save()
    return pd.DataFrame(results)


def loss(filename, ex_no: int, variant: str, T: int = 100, nreps=5_000, seed: int = 0):
    """Simplified model selection experiment computing losses by alpha

    Args:
        filename: output file
        ex_no:    experiment number
        variant:  variant (hard/easy)
        T:        length of data
        nreps:    number of monte carlo simulations to run
        seed:     random seed
    """
    ex = make_simplified_experiment(ex_no, variant)
    model_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
    y_keys = jax.random.split(y_key, nreps)
    schemes = [
        LOOCVScheme(T),
        HVBlockCVScheme(T, h=3, v=0),
        HVBlockCVScheme(T, h=3, v=3),
        KFoldCVScheme(T, k=5),
        KFoldCVScheme(T, k=10),
        LFOCVScheme(T, h=0, v=0, w=10),
        LFOCVScheme(T, h=3, v=3, w=10),
    ]
    results = []
    for alpha in ALPHA_RANGE:
        for scheme in schemes:
            ei = ex.make_simplified_instance(alpha, prng_key=model_key, model_params=dict())
            ys = jax.vmap(ei.dgp.simulate)(y_keys)
            # utility of a particular posterior with respect to the true dgp
            util_m1 = jax.vmap(lambda y: ei.mA.full_data_post(y).eljpd(ei.dgp))(ys)
            util_m2 = jax.vmap(lambda y: ei.mB.full_data_post(y).eljpd(ei.dgp))(ys)
            # CV-based model selection statistic
            sel_stats = jax.vmap(lambda y: ei.mA.cv(y, scheme) - ei.mB.cv(y, scheme))(ys)
            # note m1 clearly better
            # compute overall losses
            zeroone = jnp.mean(sel_stats < 0.0)
            # "lost log utility" from choosing m2
            logloss = jnp.mean((sel_stats < 0.0) * (util_m1 - util_m2))
            res = {
                "alpha": alpha,
                "scheme": scheme.name(),
                "zeroone": float(zeroone),
                "logloss": float(logloss),
            }
            results.append(res)
        df = pd.DataFrame(results)
        df.to_csv(filename)


def supplementary(filename, t: int = 100, seed: int = 0):
    """Run supplementary experiments"""
    model_key, sim_key = jax.random.split(jax.random.PRNGKey(seed))
    results = []
    for i in range(1,6+1):
        for variant in ['hard', 'easy']:
            print(f"Experiment {i} '{variant}' variant")
            ex = make_simplified_experiment(i, variant)
            for alpha in ALPHA_SM:
                ei = ex.make_simplified_instance(alpha, prng_key=model_key, model_params=dict())
                cv_schemes = [
                    LOOCVScheme(t),
                    HVBlockCVScheme(t, h=3, v=3),
                ]
                for sch in cv_schemes:
                    cv = ei.mA.eljpdhat_cv(sch) - ei.mB.eljpdhat_cv(sch)
                    qs, negshare = cv.sim_quantiles_neg_share(qs=QUANTILES, n=10_000, rng_key=sim_key)
                    res = dict(
                        ex_no=i,
                        variant=variant,
                        alpha=alpha,
                        scheme=sch.name(),
                        cv_mean=cv.mean(),
                        cv_std=cv.std(),
                        lower_q=qs[0],
                        upper_q=qs[-1],
                        negshare=negshare,
                    )
                    results.append(res)
                    pd.DataFrame(results).to_csv(filename)
