from functions import *
import os

def DBVSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("DBVSR")
    run_docker("DBVSR", "dbvsr", in_path,  os.path.join(os.path.expanduser("~"), "__SR_models__/DBVSR/result"), gpu, root=False, skip_frames=True, time_file=time_csv)
    move_frames("DBVSR", "result", out_path)
    add_missing_frames(out_path, in_path)


def TMNet(in_path, out_path, gpu, time_csv=None):
    clone_repository("TMNet")
    run_docker("TMNet", "tmnet", in_path, out_path, gpu, root=True, time_file=time_csv)


def SOFVSR(in_path, out_path, gpu, degradation='BI', time_csv=None):
    if degradation == 'BI':
        clone_repository("SOF-VSR-BI")
    elif degradation == 'BD':
        clone_repository("SOF-VSR-BD")
    else:
        print(degradation, "- wrong degradation")
        return None

    if degradation == 'BI':
        run_docker("SOF-VSR-BI", "sof-vsr", in_path, out_path, gpu, skip_frames=True, time_file=time_csv)
    else:
        run_docker("SOF-VSR-BD", "sof-vsr", in_path, out_path, gpu, skip_frames=True, time_file=time_csv)

    add_missing_frames(out_path, in_path)


def SOF_VSR_BD(in_path, out_path, gpu, time_csv=None):
    SOFVSR(in_path, out_path, gpu, "BI", time_csv)


def SOF_VSR_BI(in_path, out_path, gpu, time_csv=None):
    SOFVSR(in_path, out_path, gpu, "BD", time_csv)


def LGFN(in_path, out_path, gpu, time_csv=None):
    clone_repository('LGFN')
    run_docker("LGFN", "lgfn", in_path, out_path, gpu, root=False, skip_frames=True, time_file=time_csv)
    add_missing_frames(out_path, in_path)


def BasicVSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("BasicVSR")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=1EVMaV8-c2Q1r10N-CAuobIsbeSR-wptM -O ~/__SR_models__/BasicVSR/experiments/pretrained_models/BasicVSR_REDS4.pth")
    run_docker("BasicVSR", "basicvsr", in_path, out_path, gpu, root=False, time_file=time_csv)


def RSDN(in_path, out_path, gpu, time_csv=None):
    clone_repository("RSDN")
    run_docker("RSDN", "rsdn", in_path, out_path, gpu, time_file=time_csv)


def RBPN(in_path, out_path, gpu, time_csv=None):
    clone_repository("RBPN")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=11_4rsGOfbiAxqAoDoRq4vHMcXxRqc6Cc -O ~/__SR_models__/RBPN/weights/RBPN_4x.pth")
    run_docker("RBPN", "rbpn", in_path, out_path, gpu, root=False, time_file=time_csv)


def iSeeBetter(in_path, out_path, gpu, time_csv=None):
    clone_repository("iSeeBetter")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=1ROADZavabsQTX8Mc8R4GWNZ7eIwGoS_n -O ~/__SR_models__/iSeeBetter/weights/RBPN_4x.pth")
    run_docker("iSeeBetter", "iseebetter", in_path, out_path, gpu, root=True, time_file=time_csv)


def EGVSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("EGVSR")
    run_command("wget --no-check-certificate https://drive.google.com/uc?export=download&id=1CnWavBgBim6-oRFWD6oCzUEesps44bEG -O ~/__SR_models__/EGVSR/pretrained_models/EGVSR_iter420000.pth")
    run_docker("EGVSR", "egvsr", in_path, os.path.join(os.path.expanduser('~'), "__SR_models__/EGVSR/result"), gpu, root=False, time_file=time_csv)
    move_frames("EGVSR", "result", out_path)


def RealSR(in_path, out_path, gpu, time_csv=None):
    clone_repository("RealSR")
    run_docker("RealSR", "realsr", in_path, out_path, gpu, root=False, time_file=time_csv)


def Real_ESRGAN(in_path, out_path, gpu, time_csv=None):
    clone_repository("Real-ESRGAN")
    run_docker("Real-ESRGAN", "real-esrgan", in_path, out_path, gpu, root=False, time_file=time_csv)


def SwinIR(in_path, out_path, gpu, time_csv=None):
    clone_repository("Real-ESRGAN")
    run_docker("Real-ESRGAN", "swinir", in_path, out_path, gpu, root=False, time_file=time_csv)
