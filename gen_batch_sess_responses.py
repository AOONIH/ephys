from aggregate_ephys_funcs import load_or_generate_event_responses, run_decoding, parse_args

# Load plot config if provided
if __name__ == '__main__':
    load_or_generate_event_responses(parse_args())