from app.sheets_logger import SheetsLogger


class Sheet:
    def __init__(self):
        self.rows = []

    def append_row(self, row, **kwargs):
        self.rows.append(row)


def test_mesh_strategy_appended_after_existing_columns(monkeypatch):
    sheet = Sheet()
    logger = SheetsLogger("unused", "unused")
    logger._sheet = sheet
    monkeypatch.setattr(logger, "_connect", lambda: None)

    logger.append_result(
        job_name="car", project_id="project", simulation_id="simulation",
        force_results={"wetted_area": 10.0, "cd_w": 1.0},
        wind_speed=24.59, wind_direction=(-1, 0, 0),
        frontal_area=1.0, convergence_info={"status": "DRAG_SETTLED", "iterations": 5000},
        model_settings={"mesh_mode": "FIXED_MANUAL"},
    )

    assert len(sheet.rows[0]) == len(SheetsLogger.HEADERS)
    assert SheetsLogger.HEADERS[-2:] == ["Luminary Link", "Mesh Strategy"]
    assert sheet.rows[0][-1] == "FIXED_MANUAL"
    assert sheet.rows[0][32] == 5000
    assert sheet.rows[0][SheetsLogger.HEADERS.index("Wetted Area (m²)")] == 10.0
    assert sheet.rows[0][SheetsLogger.HEADERS.index("CdW (Cd × Wetted Area)")] == 1.0
