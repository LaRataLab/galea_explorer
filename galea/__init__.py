"""galea_explorer: load, quality-check and browse Galea / OpenBCI GUI recordings."""
from .io import Session, find_sessions, load_session, resolve_data  # noqa: F401
from .annotations import analysis_range, clear_valid_range, set_valid_range, valid_range  # noqa: F401
from .qc import channel_quality, channel_table, qc_report, settle_info, suggest_valid_range  # noqa: F401
