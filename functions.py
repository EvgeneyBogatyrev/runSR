import os
from pathlib import Path
from datetime import datetime
import subprocess
from multiprocessing import Process
import signal


def format_path(path):
    if '~' in path:
        path = path.replace('~', os.path.expanduser('~'))
    else:
        path = os.path.abspath(path)
    return path


def print_progress(model_path, input_videos, out_path, skip_frames, time_file=None):
    missed_frames = 0
    if skip_frames:
        missed_frames = 2

    number_of_frames = {}
    runtime = {}

    input_video_names = []
    for video in input_videos:
        video_name = list(video.split('/'))[-1]
        input_video_names.append(video_name)
        number_of_frames[video_name] = len(os.listdir(video))
    input_videos = input_video_names[:]  

    total_videos = len(number_of_frames.keys())
    current_video = None
    exhausted_videos = []
    timer_start = False
    start_num_frames = 0

    while True:
        if not os.path.exists(f"{out_path}"):
            continue

        videos = os.listdir(f"{out_path}")

        for video in videos:
            if (video not in exhausted_videos or len(exhausted_videos) == total_videos) and video in input_videos:
                if current_video is not None:
                    current_frames = min(len(os.listdir(f"{out_path}/{current_video}")), number_of_frames[current_video])
                    if current_frames >= number_of_frames[current_video] - missed_frames:
                        runtime[current_video] = datetime.now() - timer
                        print(f"{current_video} : {current_frames}/{number_of_frames[current_video]}\n", end='\r')
                        if time_file is not None:
                            process_time(list(model_path.split('/'))[-1], current_video, runtime[current_video], number_of_frames[current_video] - start_num_frames, time_file)
                        if len(exhausted_videos) == total_videos:
                            return
                    else:
                        break

                timer = datetime.now()
                current_video = video
                exhausted_videos.append(video)
                timer_start = False
                break

        if current_video is not None:
            if not timer_start and len(os.listdir(f"{out_path}/{current_video}")) > 0:
                timer_start = True
                timer = datetime.now()
                start_num_frames = len(os.listdir(f"{out_path}/{current_video}"))
            current_frames = len(os.listdir(f"{out_path}/{current_video}"))
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
    run_command(f"rm -rf ~/__SR_models__/{model}/result")
    print(f"Cloning repository to ~/__SR_models__/{model}... Done!")


def run_docker(model, image_name, in_paths, out_path, gpu, root=False, skip_frames=False, time_file=None):
    print("Running SR model...\n")
    addition = ""
    if not root:
        addition = "--user $(id -u):$(id -g) "

    mount = ""
    for path in in_paths:
        name = os.path.basename(os.path.normpath(path))
        mount += f"-v {path}:/dataset/{name} "

    command = f"docker run -it -v ~/__SR_models__/{model}:/model -v {out_path}:/results {mount} --shm-size=8192mb " + addition + f"--gpus device={gpu} --rm {image_name}"

    if image_name == "swinir":
        p = Process(target=print_progress, args=[os.path.join(os.path.expanduser('~'), f"__SR_models__/{model}/SwinIR"), in_paths, out_path, skip_frames, time_file])
    else:
        p = Process(target=print_progress, args=[os.path.join(os.path.expanduser('~'), f"__SR_models__/{model}"), in_paths, out_path, skip_frames, time_file])
    p.start()
    
    run_command(command)
    
    os.kill(p.pid, signal.SIGTERM)
    p.join()

    print("")


def move_frames(model, subdir, out_path):
    print(f"Moving results to {out_path}...", end='\r')
    videos = os.listdir(os.path.join(os.path.expanduser('~'), f"__SR_models__/{model}/{subdir}"))
    for video in videos:
        run_command(f"rm -r {out_path}/{video}")
        run_command(f"mv ~/__SR_models__/{model}/{subdir}/{video} {out_path}/{video}")
    #run_command(f"rm -r ~/__SR_models__/{model}/{subdir}")
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
