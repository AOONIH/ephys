from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
import argparse
import yaml
import platform
import os
from ephys_analysis_funcs import posix_from_win
from spikeinterface import comparison as sc, widgets as sw
from datetime import datetime
import functools
import pandas as pd
from postprocessing_utils import get_sorting_dirs, get_sorting_objs, get_waveforms,postprocess
from spikeinterface.postprocessing import (compute_spike_amplitudes, compute_correlograms,compute_template_similarity,
                                           compute_unit_locations)
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.exporters import export_report
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge, CurationSorting


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sess_date')
    args = parser.parse_args()
    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    date_str = datetime.strptime(args.sess_date,'%y%m%d').strftime('%Y-%m-%d')

    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    assert ephys_dir.is_dir()

    # sorting_dirs = get_sorting_dirs(ephys_dir,date_str, 'sorting_no_si_drift', 'kilosort2_5_no_si_drift')
    sorting_dirs = get_sorting_dirs(ephys_dir, date_str, 'sorting_no_si_drift', 'kilosort2_5_ks_drift')
    # sorting_dirs = [ceph_dir/ posix_from_win(r'X:\Dammy\ephys\DO79_240219_concat\sorting_no_si_drift\kilosort2_5_ks_drift\si_output')]
    # sorting_dirs = sorting_dirs[:1]  #  TEMP DELETE
    sorter_outputs, recordings = get_sorting_objs(sorting_dirs)

    print(f'generating waveform extractors')
    # waveforms = [si.extract_waveforms(recording, sorting,sort_dir.parent/'waveforms',allow_unfiltered=True,)
    #              if not (sort_dir.parent/'waveforms').is_dir() else si.load_waveforms(sort_dir.parent/'waveforms')
    #              for sorting, recording,sort_dir in zip(sorter_outputs, recordings, sorting_dirs)]

    [postprocess(recording, sorting, sort_dir)
     for sorting, recording, sort_dir in zip(sorter_outputs, recordings, sorting_dirs)]
    # waveforms = [get_waveforms(recording, sorting,sort_dir.parent/'waveforms')
    #              for sorting, recording,sort_dir in zip(sorter_outputs, recordings, sorting_dirs)]
    #
    # for we, sort_dir in zip(waveforms, sorting_dirs):
    #     print(f'processing {sort_dir}')
    #     _ = compute_spike_amplitudes(waveform_extractor=we)
    #     _ = compute_correlograms(we)
    #     _ = compute_unit_locations(waveform_extractor=we)
    #     _ = compute_quality_metrics(waveform_extractor=we, metric_names=['snr', 'isi_violation', 'presence_ratio'])
    #     _ = compute_template_similarity(we)
    #     export_report(waveform_extractor=we, output_folder=sort_dir.parent/'si_report',remove_if_exists=True,format='svg',
    #                   n_jobs=os.cpu_count()-1)
    #
    #
    #
    # # Automatic curation
    # for sorting, recording,we,sort_dir in zip(sorter_outputs,recordings,waveforms,sorting_dirs):
    #     merges, extra = get_potential_auto_merge(we,extra_outputs=True)
    #     if merges:
    #         clean_sorting = MergeUnitsSorting(parent_sorting=sorting,units_to_merge=merges)
    #         # we = (si.extract_waveforms(recording, clean_sorting, sort_dir.parent / 'waveforms_auto_merge',
    #         #                            allow_unfiltered=True
    #         #                            if not (sort_dir.parent / 'waveforms_auto_merge').is_dir() else si.load_waveforms(
    #         #                                sort_dir.parent / 'waveforms_auto_merge')))
    #         we = get_waveforms(recording=recording, sorting=clean_sorting, we_dir=sort_dir.parent / 'waveforms_auto_merge')
    #         export_report(waveform_extractor=we, output_folder=sort_dir.parent / 'si_report_auto_merge',
    #                       remove_if_exists=True,
    #                       format='svg', n_jobs=os.cpu_count() - 1)
    #     else:
    #         clean_sorting = sorting
    #
    #     _ = compute_spike_amplitudes(waveform_extractor=we)
    #     _ = compute_correlograms(we)
    #     _ = compute_unit_locations(waveform_extractor=we)
    #     _ = compute_template_similarity(we)
    #     cm = compute_quality_metrics(waveform_extractor=we, metric_names=['snr', 'isi_violation',
    #                                                                       'presence_ratio','firing_rate'])
    #
    #     filtered_units_df = cm.query('presence_ratio > 0.8 & (snr>=5 or isi_violations_ratio < 0.1)')
    #     filtered_units_df.index.to_frame().to_csv(sort_dir.parent / 'good_units.csv',index=False)
    #     sorting_good_units = clean_sorting.select_units(filtered_units_df.index.to_numpy())
    #     we = we.select_units(filtered_units_df.index.to_list())
    #     export_report(waveform_extractor=we, output_folder=sort_dir.parent / 'si_report_good_units',
    #                   remove_if_exists=True,
    #                   format='svg', n_jobs=os.cpu_count() - 1)
    # w = get_waveforms(recording, clean_sorting, sort_dir.parent / 'waveforms_auto_merge')
    # export_report(waveform_extractor=w, output_folder=sort_dir.parent / 'si_report_auto_merge',
    #               remove_if_exists=True,
    #               format='svg', n_jobs=os.cpu_count() - 1)
    # segment_info = pd.read_csv(sort_dir.parent.parent/'preprocessed' / 'segment_info.csv')
    # split_recordings,split_sortings = [[obj.frame_slice(start, start+n_frames)
    #                                    for start, n_frames in zip(np.cumsum(np.pad(segment_info['n_frames'], [1, 0])),
    #                                               segment_info['n_frames'])] for obj in (recording,sorting)]