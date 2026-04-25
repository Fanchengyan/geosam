"""Runtime backend selection for :mod:`geosam`.

This module stores the active runtime integration. The default runtime is the
native Python backend, while QGIS integrations can opt in explicitly from the
plugin before running CRS, logging, or progress-sensitive workflows.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

BackendName = Literal["native", "qgis", "auto"]


class ProgressBackend(Protocol):
    """Progress reporting interface shared by runtime backends."""

    def set_progress(self, percent: float) -> None:
        """Update task progress.

        Parameters
        ----------
        percent : float
            Progress percentage in the inclusive range ``0`` to ``100``.

        """

    def is_canceled(self) -> bool:
        """Return whether the active task has been canceled."""

    def push_info(self, message: str) -> None:
        """Report an informational message."""

    def push_warning(self, message: str) -> None:
        """Report a warning message."""


class NullProgressBackend:
    """No-op progress backend used outside host applications."""

    def set_progress(self, percent: float) -> None:
        """Ignore progress updates."""

    def is_canceled(self) -> bool:
        """Return ``False`` because no external task controls cancellation."""
        return False

    def push_info(self, message: str) -> None:
        """Ignore informational messages."""

    def push_warning(self, message: str) -> None:
        """Ignore warning messages."""


class QgisProgressBackend:
    """Progress backend for QGIS task and processing feedback objects."""

    def __init__(
        self,
        *,
        feedback: object | None = None,
        task: object | None = None,
    ) -> None:
        """Initialize the QGIS progress backend.

        Parameters
        ----------
        feedback : object | None, optional
            ``QgsProcessingFeedback``-compatible object.
        task : object | None, optional
            ``QgsTask``-compatible object.

        """
        self.feedback = feedback
        self.task = task

    def set_progress(self, percent: float) -> None:
        """Update QGIS task and feedback progress."""
        bounded_percent = max(0.0, min(100.0, float(percent)))
        if self.feedback is not None and hasattr(self.feedback, "setProgress"):
            self.feedback.setProgress(bounded_percent)
        if self.task is not None and hasattr(self.task, "setProgress"):
            self.task.setProgress(bounded_percent)

    def is_canceled(self) -> bool:
        """Return whether QGIS feedback or task has been canceled."""
        if (
            self.feedback is not None
            and hasattr(self.feedback, "isCanceled")
            and bool(self.feedback.isCanceled())
        ):
            return True
        if self.task is not None and hasattr(self.task, "isCanceled"):
            return bool(self.task.isCanceled())
        return False

    def push_info(self, message: str) -> None:
        """Report an informational message to QGIS feedback."""
        if self.feedback is not None and hasattr(self.feedback, "pushInfo"):
            self.feedback.pushInfo(message)

    def push_warning(self, message: str) -> None:
        """Report a warning message to QGIS feedback."""
        if self.feedback is not None and hasattr(self.feedback, "pushWarning"):
            self.feedback.pushWarning(message)


@dataclass
class RuntimeContext:
    """Active runtime integration settings.

    Attributes
    ----------
    backend : BackendName
        Selected backend name.
    qgis_project : object | None
        Optional ``QgsProject`` instance used by QGIS CRS transforms.
    qgis_transform_context : object | None
        Optional ``QgsCoordinateTransformContext``.
    qgis_feedback : object | None
        Optional ``QgsProcessingFeedback``-compatible progress sink.
    qgis_task : object | None
        Optional ``QgsTask``-compatible progress sink.
    progress : ProgressBackend
        Progress adapter used by long-running workflows.

    """

    backend: BackendName = "native"
    qgis_project: object | None = None
    qgis_transform_context: object | None = None
    qgis_feedback: object | None = None
    qgis_task: object | None = None
    progress: ProgressBackend | None = None

    def __post_init__(self) -> None:
        """Install a no-op progress backend when none is provided."""
        if self.progress is None:
            if (
                self.backend == "qgis"
                and (self.qgis_feedback is not None or self.qgis_task is not None)
            ):
                self.progress = QgisProgressBackend(
                    feedback=self.qgis_feedback,
                    task=self.qgis_task,
                )
            else:
                self.progress = NullProgressBackend()


_runtime_state = {"context": RuntimeContext()}


def _qgis_is_available() -> bool:
    """Return whether QGIS Python bindings can be imported."""
    try:
        import qgis.core  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def configure_runtime(
    backend: BackendName = "native",
    **kwargs: Any,
) -> RuntimeContext:
    """Configure the active GeoSAM runtime backend.

    Parameters
    ----------
    backend : {"native", "qgis", "auto"}, optional
        Backend to activate. ``"auto"`` selects ``"qgis"`` only when QGIS
        bindings are importable, otherwise it falls back to ``"native"``.
    **kwargs : Any
        Additional :class:`RuntimeContext` fields.

    Returns
    -------
    RuntimeContext
        The newly active runtime context.

    """
    selected_backend: BackendName
    if backend == "auto":
        selected_backend = "qgis" if _qgis_is_available() else "native"
    else:
        selected_backend = backend

    _runtime_state["context"] = RuntimeContext(backend=selected_backend, **kwargs)
    return _runtime_state["context"]


def get_runtime() -> RuntimeContext:
    """Return the active runtime context."""
    return _runtime_state["context"]


@contextmanager
def runtime_context(
    backend: BackendName = "native",
    **kwargs: Any,
) -> Iterator[RuntimeContext]:
    """Temporarily configure the runtime backend.

    Parameters
    ----------
    backend : {"native", "qgis", "auto"}, optional
        Temporary backend name.
    **kwargs : Any
        Additional :class:`RuntimeContext` fields.

    Yields
    ------
    RuntimeContext
        Temporary runtime context.

    """
    previous_context = get_runtime()
    try:
        yield configure_runtime(backend, **kwargs)
    finally:
        configure_runtime(
            previous_context.backend,
            qgis_project=previous_context.qgis_project,
            qgis_transform_context=previous_context.qgis_transform_context,
            qgis_feedback=previous_context.qgis_feedback,
            qgis_task=previous_context.qgis_task,
            progress=previous_context.progress,
        )
