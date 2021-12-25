import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def clone_repository(model):
    if not os.path.exists(f"~/__SR_models__/{model}"):
        Path(os.path.join(os.path.expanduser("~"), "__SR_models__")).mkdir(parents=True, exist_ok=True)
        os.system(f"git clone https://github.com/EvgeneyZ/{model}.git ~/__SR_models__/{model}")


def run_docker(model, image_name, in_path, gpu, root=False):
    addition = ""
    if not root:
        addition = "--user $(id -u):$(id -g) "

    command = f"docker run -it -v ~/__SR_models__/{model}:/model -v {in_path}:/dataset --shm-size=8192mb " + addition + f"--gpus='\"device={gpu}\"' --rm {image_name}"

    start_time = datetime.now()
    os.system(command)
    end_time = datetime.now()

    return end_time - start_time


def move_frames(model, subdir, out_path):
    videos = os.listdir(os.path.join(os.path.expanduser('~'), f"__SR_models__/{model}/{subdir}"))
    for video in videos:
        os.system(f"mv ~/__SR_models__/{model}/{subdir}/{video} {out_path}/{video}")


def DBVSR(in_path, out_path, gpu):
    clone_repository("DBVSR")
    runtime = run_docker("DBVSR", "dbvsr", in_path, gpu, root=True)
    move_frames("DBVSR", "result", out_path)
    add_missing_frames(out_path)
    return runtime


def TMNet(in_path, out_path, gpu):
    clone_repository("TMNet")
    runtime = run_docker("TMNet", "tmnet", in_path, gpu, root=True)
    move_frames("TMNet", "result", out_path)
    return runtime


def SOFVSR(in_path, out_path, gpu, degradation='BI'):
    if degradation == 'BI':
        clone_repository("SOF-VSR")
    elif degradation == 'BD':
        clone_repository("SOF-VSR-BD")
    else:
        print(degradation, "- wrong degradation")
        return None

    if degradation == 'BI':
        runtime = run_docker("SOF-VSR", "sof-vsr", in_path, gpu)
        move_frames("SOF-VSR", "results", out_path)
    else: 
        runtime = run_docker("SOF-VSR-BD", "sof-vsr", in_path, gpu)
        move_frames("SOF-VSR-BD", "results", out_path)

    add_missing_frames(out_path)
    return runtime


def LGFN(in_path, out_path, gpu):
    clone_repository('LGFN')
    runtime = run_docker("LGFN", "lgfn", in_path, gpu, root=True)
    move_frames("LGFN", "result", out_path)
    add_missing_frames(out_path)
    return runtime


def BasicVSR(in_path, out_path, gpu):
    clone_repository("BasicVSR")
    os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EVMaV8-c2Q1r10N-CAuobIsbeSR-wptM' -O ~/__SR_models__/BasicVSR/experiments/pretrained_models/BasicVSR_REDS4.pth")
    runtime = run_docker("BasicVSR", "basicvsr", in_path, gpu, root=True)
    move_frames("BasicVSR", "result", out_path)
    return runtime


def RSDN(in_path, out_path, gpu):
    clone_repository("RSDN")
    runtime = run_docker("RSDN", "rsdn", in_path, gpu)
    move_frames("RSDN", "result", out_path)
    return runtime


def RBPN(in_path, out_path, gpu):
    clone_repository("RBPN")
    os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=11_4rsGOfbiAxqAoDoRq4vHMcXxRqc6Cc' -O ~/__SR_models__/RBPN/weights/RBPN_4x.pth")
    runtime = run_docker("RBPN", "rbpn", in_path, gpu, root=True)
    move_frames("RBPN", "result", out_path)
    return runtime


def process_input(in_path):
    _, folders, files = next(os.walk(in_path))

    if len(folders) and len(files) > 0:
        print('Input path should only contain frames or folders with frames.')
        return None

    if len(folders) > 0:
        return in_path

    Path(os.path.join(os.path.expanduser("~"), "__dataset__/folder")).mkdir(parents=True, exist_ok=True)
    os.system(f"ffmpeg -i {in_path}/frame%04d.png -c copy ~/__dataset__/folder/frame%04d.png")

    return os.path.abspath(os.path.join(os.path.expanduser('~'), "__dataset__"))


def add_missing_frames(out_path):
    videos = os.listdir(out_path)
    for video in videos:
        os.system(f"ffmpeg -y -i {out_path}/{video}/frame0002.png -c copy {out_path}/{video}/frame0001.png")
        frames_num = len(os.listdir(f"{out_path}/{video}"))
        target = f"frame{str(frames_num).zfill(4)}.png"
        copy = f"frame{str(frames_num + 1).zfill(4)}.png"
        os.system(f"ffmpeg -y -i {out_path}/{video}/{target} -c copy {out_path}/{video}/{copy}")


def process_time(runtime, out_file, model_name, in_path):
    total_frames = 0
    _, videos, frames = next(os.walk(in_path))
    
    if len(videos) != 0:
        for video in videos:
            total_frames += len(os.listdir(f"{in_path}/{video}"))
    else:
        total_frames = len(frames)
    
    if not os.path.isfile(out_file):
        with open(out_file, 'w') as f:
            f.write('model,dataset,total_frames,total_time,fps,s/it\n')

    dataset_name = list(in_path.split('/'))[-1]
    total_time = runtime.total_seconds()
    fps = round(total_frames / total_time, 3)
    s_it = round(total_time / total_frames, 3)

    with open(out_file, 'a') as f:
        f.write(model_name + "," + dataset_name + "," + str(total_frames) + "," + str(total_time) + "," + str(fps) + "," + str(s_it) + "\n")


def main():
    parser = argparse.ArgumentParser(description = "Run SR models")
    parser.add_argument('--model', type=str, help="set model name", required=True)
    parser.add_argument('--in_path', type=str, help="set path with input videos", required=True)
    parser.add_argument('--out_path', type=str, help="set path where to store output videos", required=True)
    parser.add_argument('--gpu', type=int, default=0, help="set gpu")
    parser.add_argument('--keep_model', action='store_true', help="keep model directory")
    parser.add_argument('--csv_file', type=str, help="save csv with runtime")
    
    args = parser.parse_args()

    in_path_orig = os.path.abspath(args.in_path)
    out_path = os.path.abspath(args.out_path)

    in_path = process_input(in_path_orig)
    if in_path is None:
        return
    
    Path(out_path).mkdir(parents=True, exist_ok=True)

    if args.model == "DBVSR":
        runtime = DBVSR(in_path, out_path, args.gpu)
    elif args.model == "TMNet":
        runtime = TMNet(in_path, out_path, args.gpu)
    elif args.model == "SOF-VSR-BI":
        runtime = SOFVSR(in_path, out_path, args.gpu, 'BI')
    elif args.model == "SOF-VSR-BD":
        runtime = SOFVSR(in_path, out_path, args.gpu, 'BD')
    elif args.model == "LGFN":
        runtime = LGFN(in_path, out_path, args.gpu)
    elif args.model == "BasicVSR":
        runtime = BasicVSR(in_path, out_path, args.gpu)
    elif args.model == "RSDN":
        runtime = RSDN(in_path, out_path, args.gpu)
    elif args.model == "RBPN":
        runtime = RBPN(in_path, out_path, args.gpu)

    if in_path_orig != in_path:
        os.system(f"rm -r {in_path}")
        os.system(f"ffmpeg -i {out_path}/folder/frame%04d.png -c copy {out_path}/frame%04d.png")
        os.system(f"rm -r {out_path}/folder")

    if args.csv_file is not None and runtime is not None:
        process_time(runtime, args.csv_file, args.model, in_path_orig)

    if not args.keep_model:
        os.system(f"rm -r ~/__SR_models__")

if __name__ == "__main__":
    main()
