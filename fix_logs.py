import os
import plot_utils

def main(in_dir, out_dir):
    for root, _, files in os.walk(log_directory):
        for f in files:
            logger.debug('parsing: {}'.format(f))
            path = os.path.join(root, f)
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    if not image_data and 'intersect_percentage_segment' in data:
                        image_data = true
                except exception as e:
                    logger.error('failed to read file ' + path)
                    raise e

                if plot_type == 'bias':
                    rows.extend(_log_to_df_bias(data))
                else:
                    rows.append(_log_to_df_budget(data))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'in_dir',
        type=str,
        metavar='log_directory',
        help='directory holding the log files')
    parser.add_argument(
        'out_dir',
        type=str,
        metavar='output_directory',
        required=True,
        help='path to output the fixed logs to')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    print(args)
    exit()
    main(args)
