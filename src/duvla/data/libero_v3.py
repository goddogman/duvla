"""Read-only adapter for the official LIBERO LeRobot v3 layout.

Images are intentionally represented as video references in this first
adapter.  Decoding, resizing, camera transforms, and processor-specific
normalization belong to later, separately tested stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Frame, action_chunk, validate_episode, validate_frame


class LiberoV3Error(RuntimeError):
    """Raised when the expected LIBERO v3 files or metadata are invalid."""


@dataclass(frozen=True)
class VideoFrameRef:
    """A camera frame reference that can be decoded by a later component."""

    path: Path
    timestamp: float
    camera: str


@dataclass(frozen=True)
class LiberoSample:
    """One policy sample with an action chunk and its validity mask."""

    image: VideoFrameRef
    wrist_image: VideoFrameRef
    instruction: str
    state: tuple[float, ...]
    action_chunk: tuple[tuple[float, ...], ...]
    action_mask: tuple[bool, ...]
    timestamp: float
    frame_index: int
    episode_index: int
    task_index: int


@dataclass(frozen=True)
class _EpisodeMeta:
    image_path: Path
    wrist_image_path: Path
    instruction: str


def _as_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LiberoV3Error(f"{name} must be an integer") from exc


def _video_path(root: Path, camera: str, chunk: Any, file_index: Any) -> Path:
    return root / "videos" / camera / f"chunk-{_as_int(chunk, name='video chunk'):03d}" / (
        f"file-{_as_int(file_index, name='video file'):03d}.mp4"
    )


class LiberoV3Dataset:
    """Load LIBERO v3 rows and expose fixed-horizon policy samples.

    The adapter loads the Parquet columns into memory.  The current
    ``libero_spatial`` subset is small enough for this audit-stage loader; a
    row-group/streaming implementation should be added before larger suites.
    """

    def __init__(self, root: str | Path, *, horizon: int = 8) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        candidate = Path(root)
        # snapshot_download with local_dir can preserve the repository name as
        # one extra directory.  Accept both the outer download directory and
        # the actual LeRobot dataset root without changing any data paths.
        if not (candidate / "data").is_dir() and (candidate / "libero_spatial" / "data").is_dir():
            candidate = candidate / "libero_spatial"
        self.root = candidate
        self.horizon = horizon
        self._frames: tuple[Frame, ...]
        self._episodes: dict[int, _EpisodeMeta]
        self._episode_positions: dict[int, tuple[int, ...]]
        self._local_positions: dict[int, int]
        self._load()

    def _load(self) -> None:
        try:
            import pyarrow.parquet as parquet
        except ModuleNotFoundError as exc:
            raise LiberoV3Error(
                "pyarrow is required for LiberoV3Dataset; install it in the active environment"
            ) from exc

        if not self.root.is_dir():
            raise LiberoV3Error(f"dataset root does not exist: {self.root}")
        data_files = sorted((self.root / "data").rglob("*.parquet"))
        if not data_files:
            raise LiberoV3Error(f"no data Parquet files found below {self.root / 'data'}")

        rows: list[dict[str, object]] = []
        for data_file in data_files:
            rows.extend(parquet.read_table(data_file).to_pylist())
        frames_by_episode: dict[int, list[tuple[int, dict[str, object]]]] = {}
        frames: list[Frame] = []
        for row_position, row in enumerate(rows):
            frame = validate_frame(row)
            frames.append(frame)
            frames_by_episode.setdefault(frame.episode_index, []).append((row_position, row))

        positions: dict[int, tuple[int, ...]] = {}
        local_positions: dict[int, int] = {}
        for episode_index, episode_rows in frames_by_episode.items():
            validate_episode([row for _, row in episode_rows])
            episode_positions = tuple(position for position, _ in episode_rows)
            positions[episode_index] = episode_positions
            local_positions.update({position: local for local, position in enumerate(episode_positions)})

        task_file = self.root / "meta" / "tasks.parquet"
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not task_file.is_file() or not episode_files:
            raise LiberoV3Error("meta/tasks.parquet and meta/episodes/*.parquet are required")
        task_rows = parquet.read_table(task_file).to_pylist()
        instructions: dict[int, str] = {}
        for task_row in task_rows:
            task_index = _as_int(task_row.get("task_index"), name="task_index")
            instruction = task_row.get("__index_level_0__")
            if not isinstance(instruction, str) or not instruction.strip():
                raise LiberoV3Error(f"task {task_index} has no language instruction")
            instructions[task_index] = instruction

        episode_rows: list[dict[str, object]] = []
        for episode_file in episode_files:
            episode_rows.extend(parquet.read_table(episode_file).to_pylist())
        episode_by_index = {
            _as_int(row.get("episode_index"), name="episode_index"): row for row in episode_rows
        }
        episodes: dict[int, _EpisodeMeta] = {}
        for episode_index, episode_positions in positions.items():
            if episode_index not in episode_by_index:
                raise LiberoV3Error(f"missing metadata for episode {episode_index}")
            metadata = episode_by_index[episode_index]
            task_index = frames[episode_positions[0]].task_index
            instruction = instructions.get(task_index)
            if instruction is None:
                raise LiberoV3Error(f"missing language instruction for task {task_index}")
            task_values = metadata.get("tasks")
            if isinstance(task_values, list) and task_values and task_values[0] != instruction:
                raise LiberoV3Error(f"task metadata mismatch for episode {episode_index}")
            episodes[episode_index] = _EpisodeMeta(
                image_path=_video_path(
                    self.root,
                    "observation.images.image",
                    metadata.get("videos/observation.images.image/chunk_index"),
                    metadata.get("videos/observation.images.image/file_index"),
                ),
                wrist_image_path=_video_path(
                    self.root,
                    "observation.images.wrist_image",
                    metadata.get("videos/observation.images.wrist_image/chunk_index"),
                    metadata.get("videos/observation.images.wrist_image/file_index"),
                ),
                instruction=instruction,
            )
        self._frames = tuple(frames)
        self._episodes = episodes
        self._episode_positions = positions
        self._local_positions = local_positions

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def episode_task_indices(self) -> dict[int, int]:
        """Return the episode-to-task mapping used by deterministic splits."""

        return {
            episode_index: self._frames[positions[0]].task_index
            for episode_index, positions in self._episode_positions.items()
        }

    @property
    def frame_episode_indices(self) -> tuple[int, ...]:
        """Return the episode index for each row in dataset order."""

        return tuple(frame.episode_index for frame in self._frames)

    def __getitem__(self, index: int) -> LiberoSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        frame = self._frames[index]
        episode = self._episodes[frame.episode_index]
        positions = self._episode_positions[frame.episode_index]
        local_position = self._local_positions[index]
        actions = [self._frames[position].action for position in positions]
        chunk, mask = action_chunk(actions, start=local_position, horizon=self.horizon)
        return LiberoSample(
            image=VideoFrameRef(episode.image_path, frame.timestamp, "observation.images.image"),
            wrist_image=VideoFrameRef(
                episode.wrist_image_path, frame.timestamp, "observation.images.wrist_image"
            ),
            instruction=episode.instruction,
            state=frame.state,
            action_chunk=chunk,
            action_mask=mask,
            timestamp=frame.timestamp,
            frame_index=frame.frame_index,
            episode_index=frame.episode_index,
            task_index=frame.task_index,
        )
