import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def DBVSR(in_path, out_path, gpu):
    if not os.path.exists('~/DBVSR'):
        os.system("git clone https://github.com/EvgeneyZ/DBVSR.git ~/DBVSR")

    start_time = datetime.now()
    os.system(f"docker run -it -v ~/DBVSR:/model -v {in_path}:/dataset --shm-size=8192mb --gpus='\"device={gpu}\"' --rm dbvsr")
    end_time = datetime.now()

    vids = os.listdir(os.path.join(os.path.expanduser('~'), "DBVSR/result"))
    for vid in vids:
        os.system(f"mv ~/DBVSR/result/{vid} {out_path}/{vid}")

    add_missing_frames(out_path)
    return "~/DBVSR", end_time - start_time



def TMNet(in_path, out_path, gpu):
    if not os.path.exists('~/TMNet'):
        os.system("git clone https://github.com/EvgeneyZ/TMNet.git ~/TMNet")

    start_time = datetime.now()
    _, vids, _ = next(os.walk(in_path))
    for vid in vids:
        os.system(f"docker run -it -v ~/TMNet:/model -v {in_path}/{vid}:/dataset --shm-size=8192mb --gpus='\"device={gpu}\"' --rm tmnet")
        os.system(f"mv ~/TMNet/result/ {out_path}/{vid}")
    end_time = datetime.now()

    return "~/TMNet", end_time - start_time


def SOFVSR(in_path, out_path, gpu, degradation='BI'):
    if degradation == 'BI':
        if not os.path.exists("~/SOF-VSR"):
            os.system("git clone https://github.com/EvgeneyZ/SOF-VSR ~/SOF-VSR")
    elif degradation == 'BD':
        if not os.path.exists("~/SOF-VSR-BD"):
            os.system("git clone https://github.com/EvgeneyZ/SOF-VSR ~/SOF-VSR-BD")
    else:
        print(degradation, "- wrong degradation")
        return None, None

    if degradation == 'BI':
        start_time = datetime.now()
        os.system(f"docker run -it -v ~/SOF-VSR:/model -v {in_path}:/dataset --shm-size=8192mb --user $(id -u):$(id -g) --gpus='\"device={gpu}\"' --rm sof-vsr")
        directory = 'SOF-VSR'
        end_time = datetime.now()
    else: 
        start_time = datetime.now()
        os.system(f"docker run -it -v ~/SOF-VSR-BD:/model -v {in_path}:/dataset --shm-size=8192mb --user $(id -u):$(id -g) --gpus='\"device={gpu}\"' --rm sof-vsr")
        directory = 'SOF-VSR-BD'
        end_time = datetime.now()

    vids = os.listdir(os.path.join(os.path.expanduser('~'), f"{directory}/results"))
    for vid in vids:
        os.system(f"mv ~/{directory}/results/{vid} {out_path}/{vid}")

    add_missing_frames(out_path)
    return f"~/{directory}", end_time - start_time


def LGFN(in_path, out_path, gpu):
    if not os.path.exists("~/LGFN"):
        os.system("git clone https://github.com/EvgeneyZ/LGFN.git ~/LGFN")

    start_time = datetime.now()
    os.system(f"docker run -it -v ~/LGFN:/model -v {in_path}:/dataset --shm-size=8192mb --gpus='\"device={gpu}\"' --rm lgfn") 
    end_time = datetime.now()

    vids = os.listdir(os.path.join(os.path.expanduser('~'), "LGFN/result"))
    for vid in vids:
        os.system(f"mv ~/LGFN/result/{vid} {out_path}/{vid}")

    add_missing_frames(out_path)
    return "~/LGFN", end_time - start_time


def BasicVSR(in_path, out_path, gpu):
    if not os.path.exists("~/BasicVSR"):
        os.system("git clone https://github.com/EvgeneyZ/BasicVSR.git ~/BasicVSR")
        os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EVMaV8-c2Q1r10N-CAuobIsbeSR-wptM' -O ~/BasicVSR/experiments/pretrained_models/BasicVSR_REDS4.pth")

    start_time = datetime.now()
    os.system(f"docker run -it -v ~/BasicVSR:/model -v {in_path}:/dataset --shm-size=8192mb --user $(id -u):$(id -g) --gpus='\"device={gpu}\"' --rm basicvsr")
    end_time = datetime.now()

    vids = os.listdir(os.path.join(os.path.expanduser('~'), "BasicVSR/result"))
    for vid in vids:
        os.system(f"mv ~/BasicVSR/result/{vid} {out_path}/{vid}")
    
    return "~/BasicVSR", end_time - start_time


def process_input(in_path):
    _, folders, files = next(os.walk(in_path))

    if len(files) > 0:
        print('Input path should only contain folders with frames.')
        return False

    return True


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
    videos = os.listdir(in_path)
    
    for video in videos:
        total_frames += len(os.listdir(f"{in_path}/{video}"))
    
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

    in_path = os.path.abspath(args.in_path)
    out_path = os.path.abspath(args.out_path)

    if not process_input(args.in_path):
        return 
    
    Path(out_path).mkdir(parents=True, exist_ok=True)

    if args.model == "DBVSR":
        model_path, runtime = DBVSR(in_path, out_path, args.gpu)
    elif args.model == "TMNet":
        model_path, runtime = TMNet(in_path, out_path, args.gpu)
    elif args.model == "SOF-VSR-BI":
        model_path, runtime = SOFVSR(in_path, out_path, args.gpu, 'BI')
    elif args.model == "SOF-VSR-BD":
        model_path, runtime = SOFVSR(in_path, out_path, args.gpu, 'BD')
    elif args.model == "LGFN":
        model_path, runtime = LGFN(in_path, out_path, args.gpu)
    elif args.model == "BasicVSR":
        model_path, runtime = BasicVSR(in_path, out_path, args.gpu)

    if args.csv_file is not None and runtime is not None:
        process_time(runtime, args.csv_file, args.model, in_path)

    if not args.keep_model and model_path is not None:
        os.system(f"rm -r {model_path}")

if __name__ == "__main__":
    main()
