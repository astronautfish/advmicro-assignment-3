import numpy as np
from scipy.stats import norm
import w8_estimation as est 
import w8_LinearModel as lm
import w8_probit as probit
import w8_logit as logit

#########
# Tools #
#########

def current_and_next_value(df, var, value):

    # get unique values and sort
    uniques = np.sort(df[var].unique())
    if (uniques[0] == 0 and uniques[1] == 1):
        dummy_var = True

    else:
        dummy_var = False

    # assert value is in range
    if value < uniques[0] or value > uniques[-1]:
        raise ValueError(f'Value {value} is out of range [{uniques[0]}, {uniques[-1]})')
    
    if dummy_var:
        return 0, 1
    
    else:
        
        # find index of value and next value by going through list of uniques and flooring
        for i, i_v in enumerate(uniques):

            if i_v == value:
                indx = i
                break

            if i_v > value:
                indx = i - 1
                break

        indx_next = indx + 1

        return uniques[indx], uniques[indx_next]

def compute_simple_pe(x_old, x_new, theta_probit, theta_logit, compute_grads=False):

    # probit
    pe_probit = probit.predict(theta_probit, x_new) - probit.predict(theta_probit, x_old)

    # logit
    pe_logit = logit.predict(theta_logit, x_new) - logit.predict(theta_logit, x_old)

    if not compute_grads:
        return np.array([pe_probit, pe_logit])

    else:
        g_probit = norm.pdf(x_new * theta_probit)*x_new - norm.pdf(x_old * theta_probit)*x_new
        g_logit = logit.predict(theta_logit, x_new)*(1-logit.predict(theta_logit, x_new))*x_new - logit.predict(theta_logit, x_old)*(1 - logit.predict(theta_logit, x_old))*x_old
        return np.array([pe_probit, pe_logit]), np.array([g_probit, g_logit])
    
#######
# PEA #
#######

def compute_all_pe(x_, x_labels, data, b_probit, b_logit, cov_probit, cov_logit, do_delta=True):

    pea = np.zeros((x_.shape[0], 2)) # one col for probit, one for logit
    pea_se = np.zeros((x_.shape[0], 2))

    for i, var in enumerate(x_labels):

        if var == 'constant': continue

        idx = x_labels.index(var)
        x_low = x_.copy()
        x_high = x_.copy()

        x_low[idx], x_high[idx] = current_and_next_value(data, var, x_[idx])
        pea[i,:], gs = compute_simple_pe(x_low, x_high, b_probit, b_logit, compute_grads=True)

        if do_delta:

            # probit
            g_probit = gs[0]
            se_probit = np.sqrt(g_probit.T @ cov_probit @ g_probit)

            # logit
            g_logit = gs[1]
            se_logit = np.sqrt(g_logit.T @ cov_logit @ g_logit)

            pea_se[i,:] =  np.array([se_probit, se_logit])

        else:
            pea_se[i,:] = np.array([np.nan, np.nan]) # nan out
    
    return pea, pea_se

#######
# APE #
#######
    
def compute_ape(x, data, x_labels, x_labels_imp, b_lpm, b_probit, b_logit, do_check_lpm=True):

    pes = np.zeros((len(x_labels_imp), x.shape[0], 3)) + np.nan 

    for i, var in enumerate(x_labels_imp):

        idx = x_labels.index(var)

        pes_i = np.zeros((x.shape[0], 3)) + np.nan # logit, probit

        for obs in range(x.shape[0]):

            x_obs = x[obs]
            
            x_obs_high = x_obs.copy()
            x_obs_low = x_obs.copy()

            x_obs_low[idx], x_obs_high[idx] = current_and_next_value(data, var, x_obs[idx])

            pes_i[obs, :2] = compute_simple_pe(x_obs_low, x_obs_high, b_probit, b_logit)
            pes_i[obs, -1] = lm.predict(b_lpm, x_obs_high) - lm.predict(b_lpm, x_obs_low)
        
        pes[i, :, :] = pes_i
    
    APE = pes.mean(axis=1)

    if do_check_lpm: 
        for i, var in enumerate(x_labels_imp): print(f'{var} has same APE and LPM:\t{np.isclose(b_lpm[x_labels.index(var)], APE[i, -1])}')
    
    return APE

##########################
# Special PE of interest #
##########################

def compute_special_pes(spec_joe, x_labels, b_probit, b_logit):
    results = np.zeros((8,3,2)) + np.nan

    # outer loop over specifications: 2^3 = 8
    i = 0 # counter
    for smale in [0, 1]:
        for daytime in [0, 1]:
            for sbehavior in [0, 1]:

                special_joe_i = spec_joe.copy()

                special_joe_i[x_labels.index('smale')] = smale
                special_joe_i[x_labels.index('daytime')] = daytime
                special_joe_i[x_labels.index('sbehavior')] = sbehavior

                # inner loop over races: 3 -> #operations = 8*3 = 24

                # 1st case: white to black
                joe_1st_pre = special_joe_i.copy()
                joe_1st_post = special_joe_i.copy()

                joe_1st_pre[x_labels.index('shisp')], joe_1st_post[x_labels.index('shisp')] = 0,0
                joe_1st_pre[x_labels.index('sother')], joe_1st_post[x_labels.index('sother')] = 0,0

                joe_1st_pre[x_labels.index('sblack')] = 0.0
                joe_1st_post[x_labels.index('sblack')] = 1.0

                results[i,0,:] = compute_simple_pe(joe_1st_pre, joe_1st_post, b_probit, b_logit)

                # 2nd case: white to hisp
                joe_2nd_pre = special_joe_i.copy()
                joe_2nd_post = special_joe_i.copy()

                joe_2nd_pre[x_labels.index('sblack')], joe_2nd_post[x_labels.index('sblack')] = 0,0
                joe_2nd_pre[x_labels.index('sother')], joe_2nd_post[x_labels.index('sother')] = 0,0

                joe_2nd_pre[x_labels.index('shisp')] = 0.0
                joe_2nd_post[x_labels.index('shisp')] = 1.0

                results[i,1,:] = compute_simple_pe(joe_2nd_pre, joe_2nd_post, b_probit, b_logit)

                # 3rd case: white to other
                joe_3rd_pre = special_joe_i.copy()
                joe_3rd_post = special_joe_i.copy()

                joe_3rd_pre[x_labels.index('shisp')], joe_3rd_post[x_labels.index('shisp')] = 0,0
                joe_3rd_pre[x_labels.index('sblack')], joe_3rd_post[x_labels.index('sblack')] = 0,0

                joe_3rd_pre[x_labels.index('sother')] = 0.0
                joe_3rd_post[x_labels.index('sother')] = 1.0

                results[i,2,:] = compute_simple_pe(joe_3rd_pre, joe_3rd_post, b_probit, b_logit)

                # update counter
                i += 1

    return results

#############
# Bootstrap #
#############

def bootstrap_sample(y, x): 
    '''bootstrap_sample: samples a new dataset (with replacement) from the input. 
    Args. 
        y: 1-dimensional N-array
        x: (N,K) matrix 
    Returns
        tuple: y_i, x_i 
            y_i: N-array
            x_i: (N,K) matrix 
    '''
    N = y.size

    ii_boot = np.random.choice(N, size=N, replace=True) # indices of the bootstrap sample

    y_i = y[ii_boot] # selection of N rows from y 
    x_i = x[ii_boot] # selection of N rows from x 
    
    return y_i, x_i 

def boot(y, x, special_joe, data, x_labels, x_labels_imp, nboot=1000, seed=42, do_pea=False, average_joes=None):

    np.random.seed(seed)

    # initialize 
    b_probit_boot = np.zeros((nboot, x.shape[1])) + np.nan
    b_logit_boot = np.zeros((nboot, x.shape[1])) + np.nan
    b_lpm_boot = np.zeros((nboot, x.shape[1])) + np.nan

    special_pes_boot = np.zeros((nboot, 8, 3, 2)) + np.nan
    ape_boot = np.zeros((nboot, len(x_labels_imp,), 3)) + np.nan

    if do_pea:
        pea_boot = np.zeros((nboot, x.shape[1], 2)) + np.nan # 2 models: probit, logit
        pea_intensive_boot = np.zeros((nboot, x.shape[1], 2)) + np.nan # 2 models: probit, logit
        assert average_joes
        assert x_labels

    subsamples_events = np.zeros(nboot) + np.nan

    for i in range(nboot): 

        # 1. choose which individuals to draw
        y_i, x_i = bootstrap_sample(y, x)

        subsamples_events[i] = y_i.sum()
        print(f'Bootstrap iteration {i+1}/{nboot}: {subsamples_events[i]} events in sample')

        # 2. estimate and compute 
        ols_results =  lm.estimate(y_i, x_i, robust_se=True)

        theta0 = probit.starting_values(y_i, x_i)
        probit_results = est.estimate(probit.q, theta0, y_i, x_i, options={'disp': False}, cov_type=None)

        theta0 = logit.starting_values(y_i, x_i)
        logit_results = est.estimate(logit.q, theta0, y_i, x_i, options={'disp': False}, cov_type=None)

        # 2. save coefs
        b_probit_boot[i, :] = probit_results['theta']
        b_logit_boot[i, :] = logit_results['theta']
        b_lpm_boot[i, :] = ols_results['b_hat']

        # 3. compute pea and pea_intensive
        if do_pea:
            pea_boot[i, :, :], _ = compute_all_pe(average_joes[0], x_labels, data, probit_results['theta'], logit_results['theta'], None, None, do_delta=False)
            pea_intensive_boot[i, :, :], _ = compute_all_pe(average_joes[1], x_labels, data, probit_results['theta'], logit_results['theta'], None, None, do_delta=False)

        # 4. compute selected PEs
        special_pes_boot[i, :, :, :] = compute_special_pes(special_joe, x_labels, probit_results['theta'], logit_results['theta'])  

        # 5. compute ape
        ape_boot[i, :] = compute_ape(x, data, x_labels, x_labels_imp, ols_results['b_hat'], probit_results['theta'], logit_results['theta'], do_check_lpm=False)

    if do_pea:
        return b_lpm_boot, b_probit_boot, b_logit_boot, special_pes_boot, pea_boot, pea_intensive_boot
    
    else:
        return b_lpm_boot, b_probit_boot, b_logit_boot, special_pes_boot, ape_boot, subsamples_events