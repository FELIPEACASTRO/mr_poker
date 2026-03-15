from __future__ import annotations

from services.release_gate_service import ReleaseGateService
from services.governance_service import GovernanceService
from services.deploy_service import DeployService


class AlphaCandidateService:
    def __init__(self, release_gate: ReleaseGateService, governance: GovernanceService, deploy: DeployService) -> None:
        self.release_gate = release_gate
        self.governance = governance
        self.deploy = deploy

    def snapshot(self, model_id: str, dataset_id: str) -> dict:
        gate = self.release_gate.evaluate_candidate(model_id, dataset_id)
        card = self.governance.build_model_card(model_id)
        deploy = self.deploy.manifest()
        blockers = list(gate['blockers'])
        if not deploy['dockerfile_present'] or not deploy['compose_present']:
            blockers.append('Packaging assets incomplete for alpha candidate.')
        return {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'alpha_candidate': len(blockers) == 0,
            'release_gate': gate,
            'model_card_path': card['path'],
            'deploy_manifest': deploy,
            'blockers': blockers,
        }
