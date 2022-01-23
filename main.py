import os
import shutil
from pathlib import Path
import argparse
import models
from functions import *

def main():
    parser = argparse.ArgumentParser(description = "Run SR models")
    parser.add_argument('--model', type=str, help="set model name", required=True)
    parser.add_argument('--in_path', type=str, help="set path with input videos", required=True)
    parser.add_argument('--out_path', type=str, help="set path where to store output videos", required=True)
    parser.add_argument('--gpu', type=int, default=0, help="set gpu")
    parser.add_argument('--keep_model', action='store_true', help="keep model directory")
    parser.add_argument('--csv_file', type=str, help="save csv with runtime")
    parser.add_argument('--video_names', action="append", nargs="+", help="set video names")

    args = parser.parse_args()

    in_path_orig = format_path(args.in_path)
    out_path = format_path(args.out_path)

    in_path = process_input(in_path_orig)
    if in_path is None:
        return

    if args.video_names is None:
        video_names = os.listdir(in_path)
    else:
        video_names = args.video_names[0]
        for name in video_names:
            if name not in os.listdir(in_path):
                print(f"There is no {name} in {in_path}")
                return

    in_paths = [os.path.join(in_path, x) for x in video_names]

    Path(out_path).mkdir(parents=True, exist_ok=True)

    print_model_info(args.model)

    model = args.model.replace("-", "_")
    try:
        getattr(models, model)(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    except AttributeError:
        print(f"No such model: {args.model}")
        return

    if in_path_orig != in_path:
        if os.path.exists(in_path):
            shutil.rmtree(in_path, ignore_errors=True)
        run_command(f"cp -a {out_path}/folder/. {out_path}/")
        if os.path.exists(f"{out_path}/folder"):
            shutil.rmtree(f"{out_path}/folder", ignore_errors=True)

    if not args.keep_model:
        print("Removing SR model...", end='\r')
        if os.path.exists(f"~/__SR_models__/{args.model}"):
            shutil.rmtree(f"~/__SR_models__/{args.model}", ignore_errors=True)
        print("Removing SR model... Done!")


if __name__ == "__main__":
    main()
