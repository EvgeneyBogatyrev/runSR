import os
from pathlib import Path
import argparse
from datetime import datetime
import subprocess
from multiprocessing import Process
import signal


def print_progress(model_path, input_path, skip_frames, time_file=None):
    missed_frames = 0
    if skip_frames:
        missed_frames = 2

    input_videos = os.listdir(input_path)
    number_of_frames = {}
    runtime = {}

    for video in input_videos:
        number_of_frames[video] = len(os.listdir(f"{input_path}/{video}"))

    total_videos = len(number_of_frames.keys())
    current_video = None
    exhausted_videos = []

    while True:
        if not os.path.exists(f"{model_path}/result"):
            continue
        
        videos = os.listdir(f"{model_path}/result")

        for video in videos:
            if video not in exhausted_videos or len(exhausted_videos) == total_videos:
                if current_video is not None:
                    current_frames = min(len(os.listdir(f"{model_path}/result/{current_video}")), number_of_frames[current_video])
                    if current_frames >= number_of_frames[current_video] - missed_frames:
                        runtime[current_video] = datetime.now() - timer
                        print(f"{current_video} : {current_frames}/{number_of_frames[current_video]}\n", end='\r')
                        if time_file is not None:
                            process_time(list(model_path.split('/'))[-1], current_video, runtime[current_video], number_of_frames[current_video], time_file)
                        if len(exhausted_videos) == total_videos:
                            return
                    else:
                        break

                timer = datetime.now()
                current_video = video
                exhausted_videos.append(video)
                break
                
        if current_video is not None:
            current_frames = len(os.listdir(f"{model_path}/result/{current_video}"))
            print(f"{current_video} : {current_frames}/{number_of_frames[current_video]}", end='\r')

    
def get_user():
    with open('__tmp', 'w') as f:
        subprocess.call(['id', '-u'], stdout=f)
    with open('__tmp', 'r') as f:
        res = f.readline()
    subprocess.run(['rm', '__tmp'], capture_output=True)
    return res.strip()


def get_group():
    with open('__tmp', 'w') as f:
        subprocess.call(['id', '-g'], stdout=f)
    with open('__tmp', 'r') as f:
        res = f.readline()
    subprocess.run(['rm', '__tmp'], capture_output=True)
    return res.strip()


def run_command(command):
    command = command.replace('~', os.path.expanduser('~'))
    if '$' in command: 
        addition = "--user $(id -u):$(id -g) "
        command = command.replace(addition, ' ')
        args = list(command.split(' '))
        args.insert(7, '--user')
        args.insert(8, f'{get_user()}:{get_group()}')
    else:
        args = list(command.split(' '))
    
    while '' in args:
        args.remove('')

    subprocess.run(args, capture_output=True)


def clone_repository(model):
    print(f"Cloning repository to ~/__SR_models__/{model}...", end='\r')
    if not os.path.exists(f"~/__SR_models__/{model}"):
        Path(os.path.join(os.path.expanduser("~"), "__SR_models__")).mkdir(parents=True, exist_ok=True)
        run_command(f"git clone https://github.com/EvgeneyZ/{model}.git ~/__SR_models__/{model}")
    run_command(f"chmod -R 0777 ~/__SR_models__/{model}")
    run_command(f"rm -r ~/__SR_models__/{model}/result")
    print(f"Cloning repository to ~/__SR_models__/{model}... Done!")


def run_docker(model, image_name, in_path, gpu, root=False, skip_frames=False, time_file=None):
    print("Running SR model...\n")
    addition = ""
    if not root:
        addition = "--user $(id -u):$(id -g) "

    command = f"docker run -it -v ~/__SR_models__/{model}:/model -v {in_path}:/dataset --shm-size=8192mb " + addition + f"--gpus device={gpu} --rm {image_name}"

    p = Process(target=print_progress, args=[os.path.join(os.path.expanduser('~'), f"__SR_models__/{model}"), in_path, skip_frames, time_file])
    p.start()

    start_time = datetime.now()
    run_command(command)
    end_time = datetime.now()

    os.kill(p.pid, signal.SIGTERM)
    p.join()

    print("")
    return end_time - start_time


def move_frames(model, subdir, out_path):
    print(f"Moving results to {out_path}...", end='\r')
    videos = os.listdir(os.path.join(os.path.expanduser('~'), f"__SR_models__/{model}/{subdir}"))
    for video in videos:
        run_command(f"rm -r {out_path}/{video}")
        run_command(f"mv ~/__SR_models__/{model}/{subdir}/{video} {out_path}/{video}")
    run_command(f"rm -r ~/__SR_models__/{model}/{subdir}")
    print(f"Moving results to {out_path}... Done!") 


def print_model_info(model):
    width = os.environ.get('COLUMNS', 80)
    print('-' * width)
    print(model)
    print('-' * width)


def process_input(in_path):
    _, folders, files = next(os.walk(in_path))

    if len(folders) and len(files) > 0:
        print('Input path should only contain frames or folders with frames.')
        return None

    if len(folders) > 0:
        return in_path

    Path(os.path.join(os.path.expanduser("~"), "__dataset__/folder")).mkdir(parents=True, exist_ok=True)
    run_command(f"ffmpeg -hide_banner -loglevel error -y -i {in_path}/frame%04d.png -c copy ~/__dataset__/folder/frame%04d.png")

    return os.path.abspath(os.path.join(os.path.expanduser('~'), "__dataset__"))


def add_missing_frames(out_path):
    print("Duplicating the first and the last frame...", end='\r')
    videos = os.listdir(out_path)
    for video in videos:
        run_command(f"ffmpeg -hide_banner -loglevel error -y -i {out_path}/{video}/frame0002.png -c copy {out_path}/{video}/frame0001.png")
        frames_num = len(os.listdir(f"{out_path}/{video}"))
        target = f"frame{str(frames_num).zfill(4)}.png"
        copy = f"frame{str(frames_num + 1).zfill(4)}.png"
        run_command(f"ffmpeg -hide_banner -loglevel error -y -i {out_path}/{video}/{target} -c copy {out_path}/{video}/{copy}")
    print("Duplicating the first and the last frame... Done!")


def process_time(model_name, video, runtime, number_of_frames, out_file):
    
    if not os.path.isfile(out_file):
        with open(out_file, 'w') as f:
            f.write('model,dataset,total_frames,total_time,fps,s/it\n')

    total_time = runtime.total_seconds()
    fps = round(number_of_frames / total_time, 3)
    s_it = round(total_time / number_of_frames, 3)
    total_time = round(total_time, 3)

    with open(out_file, 'a') as f:
        f.write(model_name + "," + video + "," + str(number_of_frames) + "," + str(total_time) + "," + str(fps) + "," + str(s_it) + "\n")


def DBVSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("DBVSR")
    runtime = run_docker("DBVSR", "dbvsr", in_path, gpu, root=True, skip_frames=True, time_file=time_csv)
    move_frames("DBVSR", "result", out_path)
    add_missing_frames(out_path)
    return runtime


def TMNet(in_path, out_path, gpu, time_csv=None):
    clone_repository("TMNet")
    runtime = run_docker("TMNet", "tmnet", in_path, gpu, root=True, time_file=time_csv)
    move_frames("TMNet", "result", out_path)
    return runtime


def SOFVSR(in_path, out_path, gpu, degradation='BI', time_csv=None):
    if degradation == 'BI':
        clone_repository("SOF-VSR-BI")
    elif degradation == 'BD':
        clone_repository("SOF-VSR-BD")
    else:
        print(degradation, "- wrong degradation")
        return None

    if degradation == 'BI':
        runtime = run_docker("SOF-VSR-BI", "sof-vsr", in_path, gpu, skip_frames=True, time_file=time_csv)
        move_frames("SOF-VSR-BI", "result", out_path)
    else: 
        runtime = run_docker("SOF-VSR-BD", "sof-vsr", in_path, gpu, skip_frames=True, time_file=time_csv)
        move_frames("SOF-VSR-BD", "result", out_path)

    add_missing_frames(out_path)
    return runtime


def LGFN(in_path, out_path, gpu, time_csv=None):
    clone_repository('LGFN')
    runtime = run_docker("LGFN", "lgfn", in_path, gpu, root=True, skip_frames=True, time_file=time_csv)
    move_frames("LGFN", "result", out_path)
    add_missing_frames(out_path)
    return runtime


def BasicVSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("BasicVSR")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=1EVMaV8-c2Q1r10N-CAuobIsbeSR-wptM -O ~/__SR_models__/BasicVSR/experiments/pretrained_models/BasicVSR_REDS4.pth")
    runtime = run_docker("BasicVSR", "basicvsr", in_path, gpu, root=True, time_file=time_csv)
    move_frames("BasicVSR", "result", out_path)
    return runtime


def RSDN(in_path, out_path, gpu, time_csv=None):
    clone_repository("RSDN")
    runtime = run_docker("RSDN", "rsdn", in_path, gpu, time_file=time_csv)
    move_frames("RSDN", "result", out_path)
    return runtime


def RBPN(in_path, out_path, gpu, time_csv=None):
    clone_repository("RBPN")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=11_4rsGOfbiAxqAoDoRq4vHMcXxRqc6Cc -O ~/__SR_models__/RBPN/weights/RBPN_4x.pth")
    runtime = run_docker("RBPN", "rbpn", in_path, gpu, root=True, time_file=time_csv)
    move_frames("RBPN", "result", out_path)
    return runtime


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

    print_model_info(args.model)

    if args.model == "DBVSR":
        runtime = DBVSR(in_path, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "TMNet":
        runtime = TMNet(in_path, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "SOF-VSR-BI":
        runtime = SOFVSR(in_path, out_path, args.gpu, 'BI', time_csv=args.csv_file)
    elif args.model == "SOF-VSR-BD":
        runtime = SOFVSR(in_path, out_path, args.gpu, 'BD', time_csv=args.csv_file)
    elif args.model == "LGFN":
        runtime = LGFN(in_path, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "BasicVSR":
        runtime = BasicVSR(in_path, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "RSDN":
        runtime = RSDN(in_path, out_path, args.gpu, time_csv=args.csv_file)
    elif args.model == "RBPN":
        runtime = RBPN(in_path, out_path, args.gpu, time_csv=args.csv_file)
    else:
        print(f"Wrong model: {args.model}")
        return

    
    if in_path_orig != in_path:
        run_command(f"rm -r {in_path}")
        run_command(f"ffmpeg -i {out_path}/folder/frame%04d.png -c copy {out_path}/frame%04d.png")
        run_command(f"rm -r {out_path}/folder")

    
    if not args.keep_model:
        print("Removing SR model...", end='\r')
        run_command(f"rm -r ~/__SR_models__")
        print("Removing SR model... Done!")

if __name__ == "__main__":
    main()
