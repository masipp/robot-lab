"""Results database management using JSON files."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from robot_lab.experiments.schemas import RunResult, RunMetadata, RunMetrics


class ResultsDatabase:
    """Manage experiment results using JSON files."""
    
    def __init__(self, base_dir: str = "experiments"):
        """Initialize results database.
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Main database file
        self.db_file = self.base_dir / "experiment_database.json"
        
        # Initialize database if it doesn't exist
        if not self.db_file.exists():
            self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Create a new database file."""
        db_structure = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "experiments": {}
        }
        self._save_database(db_structure)
    
    def _load_database(self) -> Dict[str, Any]:
        """Load the database from file."""
        with open(self.db_file, 'r') as f:
            return json.load(f)
    
    def _save_database(self, data: Dict[str, Any]) -> None:
        """Save the database to file."""
        data["last_updated"] = datetime.now().isoformat()
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_experiment(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        """Add a new experiment to the database.
        
        Args:
            experiment_id: Unique experiment identifier
            metadata: Experiment metadata
        """
        db = self._load_database()
        
        if experiment_id not in db["experiments"]:
            db["experiments"][experiment_id] = {
                "metadata": metadata,
                "runs": [],
                "created_at": datetime.now().isoformat()
            }
            self._save_database(db)
            logger.success(f"Added experiment: {experiment_id}")
    
    def add_run(self, experiment_id: str, run_result: RunResult) -> None:
        """Add a run result to an experiment.
        
        Args:
            experiment_id: Experiment identifier
            run_result: Complete run result
        """
        db = self._load_database()
        
        if experiment_id not in db["experiments"]:
            self.add_experiment(experiment_id, {"name": experiment_id})
        
        # Convert Pydantic model to dict
        run_dict = run_result.model_dump() if hasattr(run_result, 'model_dump') else run_result.dict()
        
        db["experiments"][experiment_id]["runs"].append(run_dict)
        self._save_database(db)
    
    def update_run_status(
        self,
        experiment_id: str,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a run.
        
        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            status: New status
            error_message: Optional error message
        """
        db = self._load_database()
        
        if experiment_id in db["experiments"]:
            for run in db["experiments"][experiment_id]["runs"]:
                if run["metadata"]["run_id"] == run_id:
                    run["metadata"]["status"] = status
                    run["metadata"]["updated_at"] = datetime.now().isoformat()
                    if error_message:
                        run["metadata"]["error_message"] = error_message
                    break
            
            self._save_database(db)
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment data or None if not found
        """
        db = self._load_database()
        return db["experiments"].get(experiment_id)
    
    def get_runs(
        self,
        experiment_id: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get runs for an experiment, optionally filtered by status.
        
        Args:
            experiment_id: Experiment identifier
            status: Optional status filter
            
        Returns:
            List of run results
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return []
        
        runs = experiment["runs"]
        
        if status:
            runs = [r for r in runs if r["metadata"]["status"] == status]
        
        return runs
    
    def get_best_run(
        self,
        experiment_id: str,
        metric: str = "final_mean_reward"
    ) -> Optional[Dict[str, Any]]:
        """Get the best run for an experiment based on a metric.
        
        Args:
            experiment_id: Experiment identifier
            metric: Metric to use for comparison
            
        Returns:
            Best run result or None
        """
        completed_runs = self.get_runs(experiment_id, status="completed")
        
        if not completed_runs:
            return None
        
        # Find run with highest metric value
        best_run = max(
            completed_runs,
            key=lambda r: r["metrics"].get(metric, float('-inf'))
        )
        
        return best_run
    
    def get_statistics(
        self,
        experiment_id: str,
        metric: str = "final_mean_reward"
    ) -> Dict[str, Any]:
        """Get aggregate statistics for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            metric: Metric to analyze
            
        Returns:
            Dictionary of statistics
        """
        completed_runs = self.get_runs(experiment_id, status="completed")
        
        if not completed_runs:
            return {}
        
        values = [r["metrics"].get(metric, 0) for r in completed_runs]
        
        import numpy as np
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs in the database.
        
        Returns:
            List of experiment IDs
        """
        db = self._load_database()
        return list(db["experiments"].keys())
    
    def export_experiment(self, experiment_id: str, output_path: str) -> None:
        """Export experiment data to a separate JSON file.
        
        Args:
            experiment_id: Experiment identifier
            output_path: Path to save the export
        """
        experiment = self.get_experiment(experiment_id)
        
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        logger.success(f"Exported experiment to {output_file}")
