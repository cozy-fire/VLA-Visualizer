from typing import Any


def load_lerobot_frame(policy_config: Any, dataset_repo: str, frame_idx: int) -> dict[str, Any]:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(dataset_repo)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    dataset = LeRobotDataset(dataset_repo, delta_timestamps=delta_timestamps)
    if frame_idx >= dataset.num_frames:
        raise IndexError(f"帧索引 {frame_idx} 越界，数据集共有 {dataset.num_frames} 帧")
    return dataset[frame_idx]