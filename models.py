from functions import *


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
        run_docker("SOF-VSR-BI", "sof-vsr", in_path, gpu, skip_frames=True, time_file=time_csv)
        move_frames("SOF-VSR-BI", "result", out_path)
    else:
        run_docker("SOF-VSR-BD", "sof-vsr", in_path, gpu, skip_frames=True, time_file=time_csv)
        move_frames("SOF-VSR-BD", "result", out_path)

    add_missing_frames(out_path)


def LGFN(in_path, out_path, gpu, time_csv=None):
    clone_repository('LGFN')
    run_docker("LGFN", "lgfn", in_path, gpu, root=True, skip_frames=True, time_file=time_csv)
    move_frames("LGFN", "result", out_path)
    add_missing_frames(out_path)


def BasicVSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("BasicVSR")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=1EVMaV8-c2Q1r10N-CAuobIsbeSR-wptM -O ~/__SR_models__/BasicVSR/experiments/pretrained_models/BasicVSR_REDS4.pth")
    run_docker("BasicVSR", "basicvsr", in_path, gpu, root=True, time_file=time_csv)
    move_frames("BasicVSR", "result", out_path)


def RSDN(in_path, out_path, gpu, time_csv=None):
    clone_repository("RSDN")
    run_docker("RSDN", "rsdn", in_path, gpu, time_file=time_csv)
    move_frames("RSDN", "result", out_path)


def RBPN(in_path, out_path, gpu, time_csv=None):
    clone_repository("RBPN")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=11_4rsGOfbiAxqAoDoRq4vHMcXxRqc6Cc -O ~/__SR_models__/RBPN/weights/RBPN_4x.pth")
    run_docker("RBPN", "rbpn", in_path, gpu, root=True, time_file=time_csv)
    move_frames("RBPN", "result", out_path)

def iSeeBetter(in_path, out_path, gpu, time_csv=None):
    clone_repository("iSeeBetter")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=1ROADZavabsQTX8Mc8R4GWNZ7eIwGoS_n -O ~/__SR_models__/iSeeBetter/weights/RBPN_4x.pth")
    run_docker("iSeeBetter", "iseebetter", in_path, gpu, root=True, time_file=time_csv)
    move_frames("iSeeBetter", "result", out_path)


