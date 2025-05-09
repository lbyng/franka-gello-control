import os

from wandb.sdk.internal.system.system_monitor import AssetInterface

# We'll use a subset of metrics


class Settings:
    _stats_pid = os.getpid()
    _stats_sample_rate_seconds = 0.1
    _stats_samples_to_average = 1
    # _stats_disk_paths = ["/".join(os.environ["RESEARCH_LIGHTNING_REPO_PATH"].split("/")[:-1])]


settings = Settings()
interface = AssetInterface
shutdown_event = None

registry = []  # [n for n in asset_registry._registry if n.__name__ in ["CPU", "GPU", "Disk", "Memory"]]
assets = [r(AssetInterface(), settings, None) for r in registry]
metrics = [m for a in assets for m in a.metrics]
for a in assets:
    a.is_available()


def get_metrics():
    for m in metrics:
        m.sample()

    out = {}
    for a in assets:
        out.update(a.metrics_monitor.aggregate())

    for m in metrics:
        m.clear()

    return out
