import pickle
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    
    ephys_pkl_dir = r"X:\Dammy\ephys_concat_pkls"
    ephys_pkl_dir = Path(ephys_pkl_dir)
    ephys_pkl_files = list(ephys_pkl_dir.glob("*.pkl"))

    bad_units = {}

    for pkl_file in ephys_pkl_files:
        with open(pkl_file, "rb") as f:
            try:
                sess_obj = pickle.load(f)
            except:
                print(f"Failed to load {pkl_file}. Skipping.")
                continue
        
        try:
            bad_units = list(sess_obj.spike_obj.bad_units)
        except:
            print(f"No bad units found in {pkl_file}.")
            continue
        bad_units[pkl_file.stem] = bad_units
        print(f"Processed {pkl_file.stem}: {len(bad_units)} bad units found.")

    bad_units_df = pd.DataFrame.from_dict(bad_units, orient='index', columns=['bad_units'])
    output_file = ephys_pkl_dir / "bad_units_summary.csv"
    bad_units_df.to_csv(output_file, index_label='session')
    print(f"Summary of bad units saved to {output_file}.")