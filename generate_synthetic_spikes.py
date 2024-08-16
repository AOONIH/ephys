import numpy as np
from scipy import stats
from quantities import s, Hz, ms
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal.windows import gaussian


def inhomogeneous_poisson(rate, bin_size,start_time=0):
    n_bins = len(rate)
    spikes = np.random.rand(n_bins) < rate * bin_size
    spike_times = (np.nonzero(spikes)[0] * bin_size)+start_time
    return spike_times


def inhomogeneous_poisson_generator(n_trials, rate, bin_size,**kwargs):
    for i in range(n_trials):
        yield inhomogeneous_poisson(rate, bin_size,**kwargs)


def generate_isi_refractory(rate, tau_ref, n_spikes):
    lam = 1/(1/rate-tau_ref)
    isi = stats.expon.rvs(scale=1./lam, size=n_spikes)
    isi = isi + tau_ref
    return isi


def raster_plot_multi(spike_times,ax:plt.Axes):
    for i, spt in enumerate(spike_times):
        ax.vlines(spt, i, i + 1)
    # ax.set_yticks([])


def gen_rate_ts(x_ser,event_t, event_dur):
    response = np.zeros_like(x_ser)

    event_loc = np.argmin(np.abs(x_ser-event_t))
    event_width = int(np.floor(event_dur*(1/np.diff(x_ser).mean())))
    event = gaussian(M=event_width,std=50)
    response[event_loc:event_loc+event_width] = event

    return response+0.1


def gen_responses(unit_rates, n_trials, x_ser, unit_time_offsets=None, trial_var=None):

    if unit_time_offsets is None:
        unit_time_offsets = np.zeros_like(unit_rates)

    if trial_var is None:
        trial_var = np.zeros(n_trials)
    event_dur = x_ser[-1]-x_ser[0]
    bin_size = np.diff(x_ser).mean()
    # print(f'{n_trials,len(unit_rates) =}')
    # return None
    single_trial_all_units = [gen_rate_ts(x_ser,offset,0.25)*unit_rate for unit_rate,offset in zip(unit_rates, unit_time_offsets)]

    all_trial_responses = [[unit*np.random.normal(loc=1,size=len(unit),scale=t_var) for unit in single_trial_all_units]
                           for ti,t_var in enumerate(trial_var)]
    all_trial_spikes = [[next(inhomogeneous_poisson_generator(1,unit,bin_size,start_time=x_ser[0])) for unit in trial]
                        for trial in all_trial_responses]
    return all_trial_spikes


def gen_patterned_unit_rates(n_units, n_types, group_noise,max_rate=20):
    template = np.random.rand(n_units)*1000
    group_unit_rates = [template+np.random.rand(n_units)*group_noise for _ in range(n_types)]
    u_min, u_max = np.min(group_unit_rates), np.max(group_unit_rates)
    normalized_rates = [((rates-u_min)/(u_max-u_min))*max_rate
                        for rates in group_unit_rates]
    print(f'{u_min,u_max = }')
    return normalized_rates


def gen_patterned_time_offsets(n_units, n_types, group_noise,max_offset=0.2):
    template = np.random.rand(n_units)*1000
    group_unit_rates = [template+np.random.rand(n_units)*group_noise for _ in range(n_types)]
    u_min, u_max = np.min(group_unit_rates), np.max(group_unit_rates)
    normalized_rates = [((rates-u_min)/(u_max-u_min))*max_offset
                        for rates in group_unit_rates]
    print(f'{u_min,u_max = }')
    return normalized_rates


def gen_patterned_copies(template:np.ndarray,noise_list:list):
    return [template+np.random.rand(len(template))*noise for noise in noise_list]


if __name__ == '__main__':

    window = np.array([-1,1])
    max_rate = 40
    bin_size = 0.002
    time = np.arange(window[0], window[1], bin_size)
    event_loc = 0

    rate = max_rate * gaussian(len(time), 100)
    rate_ts = gen_rate_ts(time,0,0.25)*40
    rate = rate_ts

    n_trials = 10
    # spike_times = list(inhomogeneous_poisson_generator(n_trials, rate, bin_size,start_time=window[0]))
    #
    unit_rates  = [10,23,11,2,5,5,20,20]
    synth_times = gen_responses(unit_rates,n_trials,time)
    #
    spike_plot = plt.subplots(figsize=(8, 2))
    # spike_plot[1].plot(time, rate, lw=2)
    # # spike_plot[1].plot(np.mean(spike_times,axis=0))
    spike_plot[1].set_ylabel('rate (Hz)')
    spike_plot[1].set_xlabel('time (s)')
    raster_plot_multi(synth_times[1], spike_plot[1])
    plt.show()
    #
    # units_max_rates = np.random.rand(10) * max_rate
    # plt.plot(time,rate_ts)
    # # plt.show()


