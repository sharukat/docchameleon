import modal
from lib.common import image, stub
from lib.config import COLOR


def run(code: str):
    print(
        f"\n{COLOR['HEADER']}📦: RUNNING IN SANDBOX{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{code}{COLOR['ENDC']}",
        sep="\n",
    )
    sb = stub.spawn_sandbox(
        "python3.11",
        "-c",
        code,
        image=image,
        gpu="T4",
        timeout=600,  # 2 minutes
    )

    sb.wait()

    if sb.returncode != 0:
        print(
            f"{COLOR['HEADER']}📦: FAILED WITH EXITCODE {sb.returncode}{COLOR['ENDC']}\n"
        )

    return sb