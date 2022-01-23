import os
import json
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description = "Run SR models")
    parser.add_argument('--options', type=str, help="set json with options", required=True)
    parser.add_argument('--gpu', type=int, default=0, help="set gpu")
    parser.add_argument('--keep_model', action='store_true', help="keep model directory")
    parser.add_argument('--csv_file', type=str, help="save csv with runtime")

    args = parser.parse_args()

    json_path = os.path.abspath(args.options)
    if not os.path.isfile(json_path):
        print(f"{json_path} does not exist")
        return

    with open(json_path, 'r') as f:
        info = json.load(f)

    for elem in info:
        models = list(set(elem["models"]))
        dataset_path = elem["dataset_path"]
        out_path = elem["out_path"]
        video_names = elem["video_names"]

        for model in models:
            current_out_path = os.path.join(out_path, model)
            command = ['python3', 'main.py', '--model', model, '--in_path', dataset_path, '--out_path', current_out_path, '--gpu', str(args.gpu)]
            if args.csv_file is not None:
                command += ['--csv_file', args.csv_file]
            if args.keep_model:
                command += ['--keep_model']
            if video_names[0] != "all":
                command += ['--video_names'] + video_names

            subprocess.run(command)

if __name__ == "__main__":
    main()
