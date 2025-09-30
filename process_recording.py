import argparse
from pathlib import Path
import spikeinterface.full as si
import spikeinterface.preprocessing as sp
from loguru import logger
from pyinspect import install_traceback

def preprocess_and_save(recording_dir, output_dir, filter_range=(300, 9000), overwrite=False):
    """
    Preprocess a recording and save it as a binary file.

    Args:
        recording_dir (Path): Path to the recording directory.
        output_dir (Path): Path to save the preprocessed recording.
        filter_range (tuple): Frequency range for bandpass filtering.
        overwrite (bool): Whether to overwrite existing preprocessed data.
    """
    recording_dir = Path(recording_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger.info(f"Loading recording from {recording_dir}")
    recording = si.read_openephys(recording_dir)

    logger.info("Applying bandpass filter...")
    recording = sp.bandpass_filter(recording, freq_min=filter_range[0], freq_max=filter_range[1])

    logger.info("Removing bad channels...")
    bad_channels, _ = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channels)

    logger.info("Applying common median reference...")
    recording = sp.common_reference(recording, reference='global', operator='median')

    if (output_dir / "preprocessed").exists() and not overwrite:
        logger.warning(f"Preprocessed data already exists in {output_dir}. Use --overwrite to overwrite.")
        return

    logger.info(f"Saving preprocessed recording to {output_dir}")
    recording.save(folder=output_dir / "preprocessed", format="binary", overwrite=overwrite)

def main():
    install_traceback()

    parser = argparse.ArgumentParser(description="Preprocess a recording and save it as a binary file.")
    parser.add_argument("recording_dir", help="Path to the recording directory.")
    parser.add_argument("output_dir", help="Path to save the preprocessed recording.")
    parser.add_argument("--filter_range", type=float, nargs=2, default=(300, 9000),
                        help="Frequency range for bandpass filtering (default: 300-9000 Hz).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed data.")
    args = parser.parse_args()

    preprocess_and_save(args.recording_dir, args.output_dir, tuple(args.filter_range), args.overwrite)

if __name__ == "__main__":
    main()
