"""
MLflow Model Registry integration for production model lifecycle management.
Handles staging, production deployments, and model versioning.
"""

import logging
from typing import Dict, Optional, List, Any
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.pyfunc
    from mlflow.entities import ViewType

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available")


class ModelRegistry:
    """
    MLflow Model Registry for managing model lifecycle.
    Handles transitions between Dev, Staging, and Production stages.
    """

    STAGES = {
        "Development": "Development",
        "Staging": "Staging",
        "Production": "Production",
        "Archived": "Archived",
    }

    def __init__(self, registry_uri: Optional[str] = None):
        """
        Initialize Model Registry.

        Args:
            registry_uri: MLflow tracking URI
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required")

        if registry_uri:
            mlflow.set_tracking_uri(registry_uri)

        self.client = mlflow.MlflowClient()
        logger.info("MLflow Model Registry initialized")

    def register_model(
        self,
        model_name: str,
        model_uri: str,
        description: str = "",
        tags: Dict[str, str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model_uri: URI of the logged model
            description: Model description
            tags: Dictionary of tags
            run_id: MLflow run ID

        Returns:
            Version number (string)
        """
        try:
            # Create model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
            except mlflow.exceptions.RestException:
                self.client.create_registered_model(
                    model_name, tags={"environment": "development"}
                )
                logger.info(f"Created new registered model: {model_name}")

            # Register new version
            model_version = mlflow.register_model(model_uri, model_name)
            version_number = model_version.version

            # Add description and tags
            if description:
                self.client.update_model_version(
                    name=model_name, version=version_number, description=description
                )

            if tags:
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version_number,
                    key="tags",
                    value=json.dumps(tags),
                )

            # Add run ID tag
            if run_id:
                self.client.set_model_version_tag(
                    name=model_name, version=version_number, key="run_id", value=run_id
                )

            logger.info(f"Registered {model_name} version {version_number}")
            return str(version_number)

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model(
        self, model_name: str, version: str, stage: str, archive_existing: bool = True
    ) -> Dict:
        """
        Transition model to a new stage.

        Args:
            model_name: Name of the model
            version: Version number
            stage: Target stage (Development, Staging, Production, Archived)
            archive_existing: Archive existing models in target stage

        Returns:
            Updated model version info
        """
        if stage not in self.STAGES.values():
            raise ValueError(
                f"Invalid stage: {stage}. Must be one of {list(self.STAGES.values())}"
            )

        try:
            # Archive existing models in target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                current_models = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                for model_version in current_models:
                    if model_version.version != version:
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=model_version.version,
                            stage="Archived",
                        )
                        logger.info(f"Archived {model_name} v{model_version.version}")

            # Transition model
            model_version = self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )

            logger.info(f"Transitioned {model_name} v{version} to {stage}")

            return {
                "model_name": model_name,
                "version": version,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            raise

    def get_model_versions(
        self, model_name: str, stage: Optional[str] = None
    ) -> List[Dict]:
        """
        Get model versions.

        Args:
            model_name: Name of the model
            stage: Filter by stage (optional)

        Returns:
            List of model version info
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")

            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "created_timestamp": v.creation_timestamp,
                    "last_updated_timestamp": v.last_updated_timestamp,
                    "description": v.description,
                    "source": v.source,
                    "run_id": v.run_id,
                }
                for v in versions
            ]

        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get current production model."""
        versions = self.get_model_versions(model_name, stage="Production")
        return versions[0] if versions else None

    def get_staging_model(self, model_name: str) -> Optional[Dict]:
        """Get current staging model."""
        versions = self.get_model_versions(model_name, stage="Staging")
        return versions[0] if versions else None

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """
        Load a model.

        Args:
            model_name: Name of the model
            version: Version number (if not provided, uses latest)
            stage: Stage (Production, Staging, etc.)

        Returns:
            Loaded model
        """
        try:
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/Production"

            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model: {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def compare_models(
        self, model_name: str, version1: str, version2: str, metric: str = "accuracy"
    ) -> Dict:
        """
        Compare two model versions.

        Args:
            model_name: Name of the model
            version1: First version number
            version2: Second version number
            metric: Metric to compare

        Returns:
            Comparison results
        """
        try:
            versions = self.get_model_versions(model_name)

            v1_info = next((v for v in versions if v["version"] == version1), None)
            v2_info = next((v for v in versions if v["version"] == version2), None)

            if not v1_info or not v2_info:
                logger.error("One or both versions not found")
                return {}

            return {
                "model_name": model_name,
                "version1": {
                    "version": version1,
                    "stage": v1_info["stage"],
                    "created": v1_info["created_timestamp"],
                },
                "version2": {
                    "version": version2,
                    "stage": v2_info["stage"],
                    "created": v2_info["created_timestamp"],
                },
                "comparison_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}

    def print_registry_status(self, model_name: str):
        """Print registry status for a model."""
        print(f"\n{'='*70}")
        print(f"MLflow Model Registry Status: {model_name}")
        print("=" * 70)

        versions = self.get_model_versions(model_name)

        if not versions:
            print("No versions found")
            return

        # Group by stage
        by_stage = {}
        for v in versions:
            stage = v["stage"]
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(v)

        for stage in ["Production", "Staging", "Development", "Archived"]:
            if stage in by_stage:
                print(f"\n{stage}:")
                for v in by_stage[stage]:
                    print(f"  v{v['version']:3s} | Created: {v['created_timestamp']}")
                    if v["description"]:
                        print(f"           Description: {v['description']}")

        print("=" * 70 + "\n")


class CanaryDeploymentManager:
    """
    Manages canary deployments with traffic splitting and rollback.
    """

    def __init__(self, model_registry: ModelRegistry):
        """
        Initialize canary deployment manager.

        Args:
            model_registry: ModelRegistry instance
        """
        self.registry = model_registry
        self.deployments = []

    def create_canary_deployment(
        self, model_name: str, new_version: str, stages: List[Tuple[int, str]] = None
    ) -> Dict:
        """
        Create canary deployment plan.

        Args:
            model_name: Name of the model
            new_version: New version to deploy
            stages: List of (traffic_percentage, duration_minutes) tuples
                   Default: [(5, 5), (25, 10), (50, 20), (100, None)]

        Returns:
            Deployment plan
        """
        if stages is None:
            stages = [(5, 5), (25, 10), (50, 20), (100, None)]

        deployment = {
            "id": f"{model_name}-canary-{datetime.now().timestamp()}",
            "model_name": model_name,
            "new_version": new_version,
            "current_version": self.registry.get_production_model(model_name),
            "stages": stages,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "metrics": {},
        }

        self.deployments.append(deployment)
        logger.info(f"Created canary deployment: {deployment['id']}")

        return deployment

    def promote_canary_stage(
        self, deployment_id: str, stage_index: int, success: bool = True
    ) -> Dict:
        """
        Promote canary deployment to next stage.

        Args:
            deployment_id: Deployment ID
            stage_index: Current stage index
            success: Whether metrics passed thresholds

        Returns:
            Updated deployment status
        """
        deployment = next(
            (d for d in self.deployments if d["id"] == deployment_id), None
        )

        if not deployment:
            logger.error(f"Deployment {deployment_id} not found")
            return {}

        if not success:
            logger.error(f"Canary stage {stage_index} failed. Initiating rollback...")
            return self.rollback_deployment(deployment_id)

        if stage_index >= len(deployment["stages"]) - 1:
            # Final stage - promote to production
            self.registry.transition_model(
                deployment["model_name"], deployment["new_version"], "Production"
            )
            deployment["status"] = "completed"
            logger.info(
                f"Canary deployment completed. {deployment['model_name']} v{deployment['new_version']} is now in production"
            )

        return deployment

    def rollback_deployment(self, deployment_id: str) -> Dict:
        """Rollback canary deployment."""
        deployment = next(
            (d for d in self.deployments if d["id"] == deployment_id), None
        )

        if not deployment:
            return {}

        # Transition back to previous version
        if deployment["current_version"]:
            self.registry.transition_model(
                deployment["model_name"],
                deployment["current_version"]["version"],
                "Production",
            )

        deployment["status"] = "rolled_back"
        logger.warning(f"Deployment {deployment_id} rolled back")

        return deployment

    def print_deployment_plan(self, deployment: Dict):
        """Print canary deployment plan."""
        print(f"\n{'='*70}")
        print("🚀 Canary Deployment Plan")
        print("=" * 70)
        print(f"Model: {deployment['model_name']}")
        print(
            f"Version: {deployment['current_version']['version']} → {deployment['new_version']}"
        )
        print(f"\nStages:")

        for i, (traffic, duration) in enumerate(deployment["stages"]):
            duration_str = f"{duration}m" if duration else "∞"
            print(f"  Stage {i+1}: {traffic:3d}% Traffic | Duration: {duration_str}")

        print("=" * 70 + "\n")
