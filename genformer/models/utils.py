import os


def ckpt_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..","..","download_ckpts")