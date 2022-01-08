import os
from pathlib import Path
import argparse
from models import *
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

    if args.model == "DBVSR":
        DBVSR(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "TMNet":
        TMNet(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "SOF-VSR-BI":
        SOFVSR(in_paths, out_path, args.gpu, 'BI', time_csv=args.csv_file)
    elif args.model == "SOF-VSR-BD":
        SOFVSR(in_paths, out_path, args.gpu, 'BD', time_csv=args.csv_file)
    elif args.model == "LGFN":
        LGFN(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "BasicVSR":
        BasicVSR(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "RSDN":
        RSDN(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "RBPN":
        RBPN(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "iSeeBetter":
        iSeeBetter(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "EGVSR":
        EGVSR(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "Real-ESRGAN":
        Real_ESRGAN(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "SwinIR":
        SwinIR(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "RealSR":
        RealSR(in_paths, out_path, args.gpu, time_csv=args.csv_file)
    else:
        print(f"Wrong model: {args.model}")
        return
    
    if in_path_orig != in_path:
        run_command(f"rm -rf {in_path}")
        run_command(f"ffmpeg -i {out_path}/folder/frame%04d.png -c copy {out_path}/frame%04d.png")
        run_command(f"rm -rf {out_path}/folder")
    
    if not args.keep_model:
        print("Removing SR model...", end='\r')
        run_command(f"rm -rf ~/__SR_models__")
        print("Removing SR model... Done!")


if __name__ == "__main__":
    main()
