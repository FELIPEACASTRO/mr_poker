from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.session_service.service import SessionConfig, SessionRunner
from services.opponent_profile_service import OpponentProfileService
from services.coach_service import CoachService


def test_opponent_profile_and_coach_report(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'profile.db'))
    session = SessionRunner(engine=engine, store=store).run_h2h(SessionConfig(num_hands=6, seed_base=7000, session_name='coach_test'))
    session_id = session['session_id']

    profiles = OpponentProfileService(store=store, engine=engine).session_profiles(session_id)
    assert profiles['session_id'] == session_id
    assert 'seat0' in profiles['profiles']
    assert profiles['profiles']['seat0']['style_tag'] in {
        'loose_aggressive', 'loose_passive', 'tight_aggressive', 'tight_passive'
    }

    report = CoachService(store=store).session_report(session_id)
    assert report['session_id'] == session_id
    assert 'study_plan' in report
    assert len(report['study_plan']) >= 1
