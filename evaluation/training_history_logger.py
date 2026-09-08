from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calculate_moving_average(
    values,
    window: int = 20,
) -> np.ndarray:
    """
    Calculate trailing moving-average return.

    """
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if array.size == 0:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    if not isinstance(
        window,
        int,
    ):
        raise TypeError(
            "window must be an integer."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    output = np.empty_like(
        array,
        dtype=np.float64,
    )

    for index in range(
        array.size
    ):

        start = max(
            0,
            index - window + 1,
        )

        output[index] = float(
            np.mean(
                array[
                    start:index + 1
                ]
            )
        )

    return output
# ============================================================
# EXTRACT HISTORY
# ============================================================

def training_history_to_dataframe(
    history,
    moving_average_window: int = 20,
) -> pd.DataFrame:
    """
    Convert FC-HMARL TrainingHistory into a pandas DataFrame.
    --------------------------------------
        """

    if history is None:
        raise ValueError(
            "history cannot be None."
        )

    if not hasattr(
        history,
        "episodes",
    ):
        raise TypeError(
            "history must contain an 'episodes' attribute."
        )

    episodes = list(
        history.episodes
    )

    if len(
        episodes
    ) == 0:
        raise ValueError(
            "Training history contains no episodes."
        )

    records: List[
        Dict[str, Any]
    ] = []

    cumulative_steps = 0

    for statistics in episodes:

        episode_number = int(
            statistics.episode
        )

        steps = int(
            statistics.steps
        )

        cumulative_steps += steps

        local_returns = np.asarray(
            statistics.local_returns,
            dtype=np.float64,
        )

        if local_returns.ndim != 1:
            raise ValueError(
                "local_returns must be one-dimensional."
            )

        if not np.isfinite(
            local_returns
        ).all():
            raise ValueError(
                "local_returns contains NaN or Inf."
            )

        coordinator_return = float(
            statistics.coordinator_return
        )

        total_return = float(
            statistics.total_return
        )

        if not np.isfinite(
            coordinator_return
        ):
            raise ValueError(
                "coordinator_return contains NaN or Inf."
            )

        if not np.isfinite(
            total_return
        ):
            raise ValueError(
                "total_return contains NaN or Inf."
            )

        record: Dict[
            str,
            Any,
        ] = {
            "episode":
                episode_number,

            "steps":
                steps,

            "cumulative_steps":
                cumulative_steps,

            "total_return":
                total_return,

            "coordinator_return":
                coordinator_return,

            "mean_local_return":
                float(
                    np.mean(
                        local_returns
                    )
                ),

            "sum_local_return":
                float(
                    np.sum(
                        local_returns
                    )
                ),

            "minimum_local_return":
                float(
                    np.min(
                        local_returns
                    )
                ),

            "maximum_local_return":
                float(
                    np.max(
                        local_returns
                    )
                ),

            "local_update_count":
                int(
                    statistics
                    .local_update_count
                ),

            "coordinator_update_count":
                int(
                    statistics
                    .coordinator_update_count
                ),

            "terminated_early":
                bool(
                    statistics
                    .terminated_early
                ),
        }

        # ----------------------------------------------------
        # Store each individual microgrid return
        # ----------------------------------------------------

        for (
            local_index,
            local_return,
        ) in enumerate(
            local_returns,
            start=1,
        ):

            record[
                f"mg{local_index}_return"
            ] = float(
                local_return
            )

        records.append(
            record
        )

    dataframe = pd.DataFrame(
        records
    )
    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    dataframe[
        "moving_average_return"
    ] = calculate_moving_average(
        dataframe[
            "total_return"
        ].to_numpy(),
        window=moving_average_window,
    )

    dataframe[
        "moving_average_coordinator_return"
    ] = calculate_moving_average(
        dataframe[
            "coordinator_return"
        ].to_numpy(),
        window=moving_average_window,
    )

    dataframe[
        "moving_average_local_return"
    ] = calculate_moving_average(
        dataframe[
            "mean_local_return"
        ].to_numpy(),
        window=moving_average_window,
    )

    # ========================================================
    # CUMULATIVE UPDATE COUNTS
    # ========================================================

    dataframe[
        "cumulative_local_updates"
    ] = (
        dataframe[
            "local_update_count"
        ]
        .cumsum()
    )

    dataframe[
        "cumulative_coordinator_updates"
    ] = (
        dataframe[
            "coordinator_update_count"
        ]
        .cumsum()
    )

    return dataframe

# ============================================================
# SAVE CSV
# ============================================================

def save_training_history_csv(
    history,
    output_path: str | Path = (
        "outputs/results/training_history.csv"
    ),
    moving_average_window: int = 20,
) -> Path:
    """
    Save complete episode-level FC-HMARL training history.
    """

    dataframe = (
        training_history_to_dataframe(
            history=history,
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path
# ============================================================
# CONVERGENCE FIGURE
# ============================================================

def save_convergence_figure(
    history,
    output_path: str | Path = (
        "outputs/results/"
        "training_convergence.png"
    ),
    moving_average_window: int = 20,
    dpi: int = 300,
) -> Path:
    """
    Save FC-HMARL episode-return convergence figure.
    """

    dataframe = (
        training_history_to_dataframe(
            history=history,
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )

    axis.plot(
        dataframe[
            "episode"
        ],
        dataframe[
            "total_return"
        ],
        linewidth=1.0,
        alpha=0.45,
        label="Episode return",
    )

    axis.plot(
        dataframe[
            "episode"
        ],
        dataframe[
            "moving_average_return"
        ],
        linewidth=2.0,
        label=(
            f"{moving_average_window}-episode "
            "moving average"
        ),
    )

    axis.set_xlabel(
        "Training Episode"
    )

    axis.set_ylabel(
        "Total Episode Return"
    )

    axis.set_title(
        "FC-HMARL Training Convergence"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path
# ============================================================
# RETURN COMPONENT FIGURE
# ============================================================

def save_return_components_figure(
    history,
    output_path: str | Path = (
        "outputs/results/"
        "training_return_components.png"
    ),
    moving_average_window: int = 20,
    dpi: int = 300,
) -> Path:
    """
    Plot mean local-agent return and coordinator return.
    """

    dataframe = (
        training_history_to_dataframe(
            history=history,
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )

    axis.plot(
        dataframe[
            "episode"
        ],
        dataframe[
            "moving_average_local_return"
        ],
        linewidth=2.0,
        label="Mean local-agent return",
    )

    axis.plot(
        dataframe[
            "episode"
        ],
        dataframe[
            "moving_average_coordinator_return"
        ],
        linewidth=2.0,
        label="Coordinator return",
    )

    axis.set_xlabel(
        "Training Episode"
    )

    axis.set_ylabel(
        "Moving-Average Return"
    )

    axis.set_title(
        "Hierarchical FC-HMARL Return Components"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path
# ============================================================
# UPDATE COUNT FIGURE
# ============================================================

def save_update_count_figure(
    history,
    output_path: str | Path = (
        "outputs/results/"
        "training_update_counts.png"
    ),
    moving_average_window: int = 20,
    dpi: int = 300,
) -> Path:
    """
    Plot cumulative local and coordinator SAC updates.
    """

    dataframe = (
        training_history_to_dataframe(
            history=history,
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )

    axis.plot(
        dataframe[
            "episode"
        ],
        dataframe[
            "cumulative_local_updates"
        ],
        linewidth=2.0,
        label="Local SAC updates",
    )

    axis.plot(
        dataframe[
            "episode"
        ],
        dataframe[
            "cumulative_coordinator_updates"
        ],
        linewidth=2.0,
        label="Coordinator SAC updates",
    )

    axis.set_xlabel(
        "Training Episode"
    )

    axis.set_ylabel(
        "Cumulative Gradient Updates"
    )

    axis.set_title(
        "FC-HMARL SAC Update Progress"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# ============================================================
# COMPLETE EXPORT FUNCTION
# ============================================================

def export_training_results(
    history,
    output_directory: str | Path = (
        "outputs/results"
    ),
    moving_average_window: int = 20,
) -> Dict[str, Path]:
    """
    Export all training-history results.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        save_training_history_csv(
            history=history,
            output_path=(
                output_directory
                / "training_history.csv"
            ),
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    convergence_path = (
        save_convergence_figure(
            history=history,
            output_path=(
                output_directory
                / "training_convergence.png"
            ),
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    components_path = (
        save_return_components_figure(
            history=history,
            output_path=(
                output_directory
                / "training_return_components.png"
            ),
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    updates_path = (
        save_update_count_figure(
            history=history,
            output_path=(
                output_directory
                / "training_update_counts.png"
            ),
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    return {
        "training_history_csv":
            csv_path,

        "training_convergence_figure":
            convergence_path,

        "return_components_figure":
            components_path,

        "update_count_figure":
            updates_path,
    }
