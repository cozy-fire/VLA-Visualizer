from typing import Any

from .images import collect_image_observations
from .serialization import serialize_images, summarize_frame_state


def load_lerobot_frame(policy_config: Any, dataset_repo: str, frame_idx: int) -> dict[str, Any]:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(dataset_repo)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    dataset = LeRobotDataset(dataset_repo, delta_timestamps=delta_timestamps)
    if frame_idx >= dataset.num_frames:
        raise IndexError(f"frame index {frame_idx} out of range; dataset has {dataset.num_frames} frames")
    return dataset[frame_idx]


def load_lerobot_frame_page(
    dataset_repo: str, page: int = 0, page_size: int = 24, stride: int = 1,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    page = max(0, int(page))
    page_size = min(100, max(1, int(page_size)))
    stride = max(1, int(stride))
    dataset = LeRobotDataset(dataset_repo)
    num_frames = int(dataset.num_frames)
    sampled_frames = 0 if num_frames <= 0 else ((num_frames - 1) // stride) + 1
    sample_start = min(page * page_size, sampled_frames)
    sample_stop = min(sample_start + page_size, sampled_frames)
    frames = []
    for sample_idx in range(sample_start, sample_stop):
        frame_idx = sample_idx * stride
        item = dataset[frame_idx]
        image_keys, images = collect_image_observations(item)
        frames.append({
            "frame_idx": frame_idx,
            "image_keys": image_keys,
            "images": serialize_images(images, max_size=220),
            "state_summary": summarize_frame_state(item),
        })
    return {
        "num_frames": num_frames,
        "sampled_frames": sampled_frames,
        "stride": stride,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (sampled_frames + page_size - 1) // page_size),
        "frames": frames,
    }