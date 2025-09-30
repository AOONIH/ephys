import shutil
from pathlib import Path

import pandas as pd
import spikeinterface.full as si
import functools
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from spikeinterface.postprocessing import (compute_spike_amplitudes, compute_correlograms,compute_template_similarity,
                                           compute_unit_locations)
from spikeinterface import create_sorting_analyzer, load_sorting_analyzer
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.exporters import export_report
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge, CurationSorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from copy import deepcopy as copy
import os


def get_sorting_dirs(ephys_dir:Path, sess_name_date, sorting_dir_name, sorter_dir_name, output_name='si_output'):
    """

    :param ephys_dir:
    :param sess_name_date:
    :param sorting_dir_name:
    :param sorter_dir_name:
    :param output_name:
    :return: sorting directories for all sessions for an animal on a day
    """
    sess_dirs = sorted(list(ephys_dir.glob(f'*{sess_name_date}*')))
    print(sess_dirs)
    sorting_dirs = [sess / sorting_dir_name / sorter_dir_name / output_name if (sess / sorting_dir_name / sorter_dir_name / output_name).is_dir()
                    else (sess / sorting_dir_name / sorter_dir_name / output_name, (sess / sorting_dir_name / sorter_dir_name / output_name).mkdir(parents=True))[0] for sess in sess_dirs]
    [Warning(f'sort dir empty for {sort_dir}') for sort_dir in sorting_dirs if not list(sort_dir.glob('*'))]
    assert all([sorting_dir.is_dir() for sorting_dir in sorting_dirs]) and bool(sorting_dirs), 'Not all paths are directories'
    # [next(e.rglob('spike_times.npy')).parent for e in sorting_dirs]
    # return [next(e.rglob('spike_times.npy')).parent for e in sorting_dirs]
    return sorting_dirs


def get_sorting_objs(sorting_dirs):
    sorter_outputs = [si.read_numpy_sorting_folder(next(folder).parent if folder.stem != 'si_output' else folder)
                      for folder in sorting_dirs]
    recording_dirs = [folder.parent.parent / 'preprocessed' for folder in sorting_dirs]
    recordings = [si.read_binary_folder(recording_dir) for recording_dir in recording_dirs]
    [recording.annotate(is_filtered=True) for recording in recordings]
    return sorter_outputs, recordings


def get_waveforms(recording, sorting, we_dir:Path, **kwargs):
    if not we_dir.is_dir() or kwargs.get('overwrite', True):
        waveforms = si.extract_waveforms(recording, sorting, we_dir,overwrite=None,**kwargs)
    else:
        try:
            waveforms = si.load_waveforms(we_dir)
        except:
            waveforms = si.extract_waveforms(recording, sorting, we_dir, overwrite=None, **kwargs)
    return waveforms


def get_qaulity_metrics(we):
    _ = compute_spike_amplitudes(waveform_extractor=we)
    _ = compute_correlograms(we)
    _ = compute_unit_locations(waveform_extractor=we)
    _ = compute_template_similarity(we)
    cm = compute_quality_metrics(waveform_extractor=we, metric_names=['snr', 'isi_violation',
                                                                      'presence_ratio', 'firing_rate'])


def map_channels2recordings(recording):
    # probes = {i: probe for i, probe in recording.get_probes()}
    # shank_ids = [probes[p].shank_ids for p in probes]
    probes = recording.get_probegroup().to_dataframe(complete=True)


def plot_periodogram(freqs,psd):
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd, color='k', lw=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    plt.xlim([0, freqs.max()])
    plt.show()


def get_probe_power(probes_df,rec_segment,fs,f_band=(0, 200)):
    y_pos = sorted(probes_df['y'].unique())
    probes_power = np.full((probes_df['shank_ids'].unique().shape[0],len(y_pos),int((f_band[1]-f_band[0])/2+1)),-1e6)

    for shank_i, in sorted(probes_df['shank_ids'].unique()):
        ids2use = probes_df.query(f'shank_ids=="{str(shank_i)}"').index
        shank_power_arr = np.array([signal.welch(rec_segment[i], fs=fs, nperseg=0.5 * fs) for i in ids2use])
        freqs = copy(shank_power_arr[0,0,:])
        shank_power_arr = shank_power_arr[:,1,:][:,freqs >= f_band[0]][:,freqs <= f_band[1]]
        freqs = freqs[freqs >= f_band[0]][freqs <= f_band[1]]
        # print(freqs,freqs.shape)
        lfp_spectrum_data = 10 * np.log(shank_power_arr)
        # print(type(shank_i))
        # print(np.in1d(y_pos,probes_df.loc[ids2use,'y']))
        probes_power[int(shank_i),np.in1d(y_pos,probes_df.loc[ids2use,'y'])] = lfp_spectrum_data
    # print(probes_power.shape)#

    probes_power[probes_power==-1e6] = np.quantile(probes_power[probes_power!=-1e6],0.01)
    # probes_power = probes_power.reshape((probes_df['probe_index'].unique().shape[0],-1,len(freqs)))

    return probes_power,freqs


def postprocess(recording, sorting, sort_dir:Path):
    sorting = sorting.remove_empty_units()
    print(f'Postprocessing {sort_dir.parent}')
    # recs_by_group = recording.split_by('shank_id')
    recs_by_group = {0:recording}
    print(f'{recs_by_group = }')
    good_units_by_group = {}
    if len(recs_by_group) > 1:
        we_dir_sffx = [f'group_{i}' for i in range(len(recs_by_group))]
    else:
        we_dir_sffx = ['']
    # for ri,rec in enumerate(recs_by_group.values()):
    we_dir = sort_dir.parent / f'waveforms'

    analyzer = create_sorting_analyzer(recording=recording, sorting=sorting, format='memory',
                                       folder=we_dir)
    sparsity = si.compute_sparsity(analyzer, method="by_property", by_property='shank_id')

    analyzer = create_sorting_analyzer(recording=recording, sorting=sorting, format='memory',
                                       folder=we_dir, sparsity=sparsity)

    analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    analyzer.compute(['spike_amplitudes', 'correlograms', 'template_similarity', 'quality_metrics'],
                     extension_params=dict(
                     quality_metrics=dict(metric_names=['snr', 'isi_violation', 'presence_ratio']),
                     ), n_jobs=16,
                     )

    # the export process
    if (sort_dir.parent / f'si_report').exists():
        shutil.rmtree(sort_dir.parent / f'si_report', ignore_errors=True)
    export_report(sorting_analyzer=analyzer, output_folder=sort_dir.parent/f'si_report',
                  remove_if_exists=True, format='png')


def get_shank_spectrum_by_depth(recording):
    fs = int(recording.sampling_frequency)
    probes = recording.get_probegroup().to_dataframe(complete=True)
    indices_by_depth = [probes.sort_values(['y','x'], ascending=False).query('probe_index==@probe_i').index for
                        probe_i in probes['probe_index'].unique()]
    traces_start = recording.get_traces(start_frame=10000, end_frame=10000 + 30 * fs).T
    traces_end = recording.get_traces(start_frame=-30 * fs, end_frame=-1).T
    y_pos = sorted(probes['y'].unique())
    plots = []
    for traces,lbl in zip([traces_start,traces_end],['start','end']):
        lfp_spectrum_data, freqs = get_probe_power(probes,traces,fs)
        shank_spectrum_plot = plt.subplots(2,4,sharey='all',sharex='all')
        for shank_i,ax in zip(sorted(probes['shank_ids'].unique()),shank_spectrum_plot[1].flatten()):
            dB_levels = np.quantile(lfp_spectrum_data, [0.1, 0.9])
            spectrum_plot = ax.imshow(lfp_spectrum_data[int(shank_i)],aspect='auto',origin='lower',
                                      extent=[freqs.min(),freqs.max(),max(y_pos),0],
                                      vmin=dB_levels[0], vmax=dB_levels[1])
        shank_spectrum_plot[0].subplots_adjust(right=0.84)
        # shank_spectrum_plot[0].subplots_adjust(wspace=0.01,hspace=0.05)
        cbar_ax = shank_spectrum_plot[0].add_axes([0.86, 0.15, 0.02, 0.7])
        cbar=shank_spectrum_plot[0].colorbar(spectrum_plot, cax=cbar_ax)
        cbar.set_label('LFP power (dB)')
        shank_spectrum_plot[0].suptitle(f'LFP power spectrum by shank: {lbl}')
        shank_spectrum_plot[0].show()
        plots.append(shank_spectrum_plot)
    return plots
