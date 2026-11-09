"""galea_explorer: load, quality-check and browse Galea / OpenBCI GUI recordings."""
from .io import Session, find_sessions, load_session, resolve_data  # noqa: F401
from .qc import channel_quality, channel_table, qc_report, settle_info, settle_time  # noqa: F401
