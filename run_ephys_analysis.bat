@echo off
SET "config_file=config.yaml"
SET "sorter_dirname=%~2"
IF "%sorter_dirname%"=="" SET "sorter_dirname=from_concat"

python H:\ephys\ephys_analysis_multisess.py %config_file% %1 --sorter_dirname "%sorter_dirname%"
