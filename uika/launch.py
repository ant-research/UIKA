import argparse
import importlib
import os


def isolate_local_cuda_device():
    """Expose only this rank's local CUDA device before torch is imported."""
    if os.environ.get("UIKA_DISABLE_CUDA_DEVICE_ISOLATION") == "1":
        return

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    try:
        local_rank_idx = int(local_rank)
    except ValueError:
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    else:
        try:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", local_rank_idx + 1))
        except ValueError:
            local_world_size = local_rank_idx + 1
        devices = [str(device_idx) for device_idx in range(local_world_size)]

    if local_rank_idx >= len(devices):
        return

    os.environ.setdefault("UIKA_ORIGINAL_LOCAL_RANK", local_rank)
    os.environ.setdefault("UIKA_ORIGINAL_CUDA_VISIBLE_DEVICES", visible_devices or "")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank_idx]
    os.environ["LOCAL_RANK"] = "0"


isolate_local_cuda_device()

from uika.runners import REGISTRY_RUNNERS


def import_runner_module(runner: str):
    module_name = f"uika.runners.{runner}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Runner module {module_name} not found") from exc
        raise


def main():

    parser = argparse.ArgumentParser(description='UIKA launcher')
    parser.add_argument('runner', type=str, help='Runner to launch')
    args, unknown = parser.parse_known_args()

    import_runner_module(args.runner)

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError('Runner {} not found'.format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    with RunnerClass() as runner:
        runner.run()


if __name__ == '__main__':
    main()
