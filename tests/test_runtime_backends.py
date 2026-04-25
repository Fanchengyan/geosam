from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

from geosam.context import configure_runtime, get_runtime
from geosam.logging import setup_logger


class FakeMessageLog:
    messages: ClassVar[list[tuple[str, str, object]]] = []

    @classmethod
    def logMessage(cls, message: str, tag: str, level: object) -> None:
        cls.messages.append((message, tag, level))


def _install_fake_qgis(monkeypatch) -> SimpleNamespace:
    fake_qgis = ModuleType("qgis")
    fake_core = ModuleType("qgis.core")
    fake_levels = SimpleNamespace(Info="info", Warning="warning", Critical="critical")
    fake_core.Qgis = SimpleNamespace(MessageLevel=fake_levels)
    fake_core.QgsMessageLog = FakeMessageLog
    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.core", fake_core)
    return fake_levels


def test_qgis_runtime_routes_logging_to_qgis_message_log(monkeypatch) -> None:
    fake_levels = _install_fake_qgis(monkeypatch)
    FakeMessageLog.messages.clear()
    configure_runtime("qgis")

    logger = setup_logger(
        "geosam.tests.qgis_logging",
        clear_existing=True,
        propagate=False,
    )
    logger.warning("Visible in QGIS")

    assert FakeMessageLog.messages
    message, tag, level = FakeMessageLog.messages[-1]
    assert "Visible in QGIS" in message
    assert tag == "GeoSAM"
    assert level == fake_levels.Warning

    configure_runtime("native")


def test_qgis_runtime_uses_feedback_progress_backend() -> None:
    class FakeFeedback:
        def __init__(self) -> None:
            self.progress: float | None = None
            self.info_messages: list[str] = []
            self.warning_messages: list[str] = []

        def setProgress(self, percent: float) -> None:
            self.progress = percent

        def isCanceled(self) -> bool:
            return False

        def pushInfo(self, message: str) -> None:
            self.info_messages.append(message)

        def pushWarning(self, message: str) -> None:
            self.warning_messages.append(message)

    feedback = FakeFeedback()
    runtime = configure_runtime("qgis", qgis_feedback=feedback)
    runtime.progress.set_progress(125.0)
    runtime.progress.push_info("done")
    runtime.progress.push_warning("careful")

    assert feedback.progress == 100.0
    assert feedback.info_messages == ["done"]
    assert feedback.warning_messages == ["careful"]
    assert get_runtime().backend == "qgis"

    configure_runtime("native")
