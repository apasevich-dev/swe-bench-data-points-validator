import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from swe_bench_validator.data_point_reader import DataPointReader
from swe_bench_validator.prediction_converter import PredictionConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Validator:    
    def __init__(self, data_dir: str = "data_points"):
        self.data_dir = Path(data_dir)
        self.loader = DataPointReader()
        self.formatter = PredictionConverter(model_name="gpt-4")
    
    def validate(self, files: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            if files is None:
                json_files = list(self.data_dir.glob("*.json"))
                files = [f.stem for f in json_files]
                logger.info(f"Processing {len(files)} files")
            else:
                logger.info(f"Processing {len(files)} specified files")
            
            if not files:
                return {"error": "No files found for processing"}
            
            results = {
                "total_files": len(files),
                "successful_files": 0,
                "failed_files": 0,
                "file_results": {}
            }
            
            for file in files:
                logger.info(f"Processing: {file}")
                file_result = self._validate_file(file)
                results["file_results"][file] = file_result
                
                if file_result.get("success", False):
                    results["successful_files"] += 1
                else:
                    results["failed_files"] += 1
                    logger.error(f"Failed: {file} - {file_result.get('error', 'unknown')}")
            
            success_rate = (results["successful_files"] / results["total_files"]) * 100 if results["total_files"] > 0 else 0
            results["success_rate"] = success_rate
            
            return results
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"error": str(e)}
    
    def _validate_file(self, file: str) -> Dict[str, Any]:
        run_id = file
        predictions_file = f"predictions_{file}.jsonl"
        logs_dir = Path("logs/run_evaluation") / run_id / "gpt-4"
        
        try:
            data_points = self.loader.load(self.data_dir, [file])
            
            if not data_points:
                return {"success": False, "error": f"Failed to load {file}.json"}
            
            data_point = data_points[0]
            instance_id = data_point.get("instance_id", "unknown")
            
            predictions = self.formatter.convert([data_point])
            
            if not predictions:
                return {"success": False, "error": "Failed to convert to prediction"}
            
            if not self.formatter.save_to_file(predictions, predictions_file):
                return {"success": False, "error": "Failed to save prediction"}
            
            docker_result = self.evaluation(predictions_file, run_id)
            
            if not docker_result["success"]:
                return {"success": False, "error": f"Docker evaluation failed: {docker_result['error']}"}
            
            validation_result = self._check_result(data_point, logs_dir)
            
            return {
                "success": True,
                "instance_id": instance_id,
                "run_id": run_id,
                "validation_result": validation_result
            }
            
        except Exception as e:
            logger.error(f"Error validating {file}: {e}")
            return {"success": False, "error": str(e)}
    
    def evaluation(self, predictions_file: str, run_id: str) -> Dict[str, Any]:        
        cmd = [
            "docker", "compose", "run", "--rm", "swe-bench-validator",
            "python", "-m", "swebench.harness.run_evaluation",
            "--predictions_path", predictions_file,
            "--run_id", run_id,
            "--dataset_name", "SWE-bench/SWE-bench",
            "--clean", "True"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": f"Exit code: {result.returncode}"}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_result(self, data_point: Dict[str, Any], logs_dir: Path) -> Dict[str, Any]:        
        instance_id = data_point["instance_id"]
        expected_fail_to_pass = json.loads(data_point["FAIL_TO_PASS"])
        expected_pass_to_pass = json.loads(data_point["PASS_TO_PASS"])
        report_path = logs_dir / instance_id / "report.json"
        
        if not report_path.exists():
            return {
                "status": "report_not_found",
                "error": f"File {report_path} not found"
            }
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            instance_report = report.get(instance_id, {})
            tests_status = instance_report.get("tests_status", {})
            
            actual_fail_to_pass = tests_status.get("FAIL_TO_PASS", {}).get("success", [])
            actual_pass_to_pass = tests_status.get("PASS_TO_PASS", {}).get("success", [])
            
            fail_to_pass_match = set(expected_fail_to_pass) == set(actual_fail_to_pass)
            pass_to_pass_match = set(expected_pass_to_pass) == set(actual_pass_to_pass)
            
            is_resolved = instance_report.get("resolved", False)
            
            if fail_to_pass_match and pass_to_pass_match and is_resolved:
                status = "success"
            else:
                status = "test_mismatch"
            
            return {
                "status": status,
                "resolved": is_resolved,
                "fail_to_pass_match": fail_to_pass_match,
                "pass_to_pass_match": pass_to_pass_match,
                "expected_fail_to_pass": expected_fail_to_pass,
                "actual_fail_to_pass": actual_fail_to_pass,
                "expected_pass_to_pass": expected_pass_to_pass,
                "actual_pass_to_pass": actual_pass_to_pass
            }
            
        except Exception as e:
            return {
                "status": "read_error",
                "error": str(e)
            }
    
    def display_results(self, results: Dict[str, Any]) -> None:        
        print("\n" + "="*60)
        print("SWE-bench Validation Results")
        print("="*60)
        
        if "error" in results:
            print(f"Validation failed: {results['error']}")
            return
        
        total_files = results.get("total_files", 0)
        successful_files = results.get("successful_files", 0)
        failed_files = results.get("failed_files", 0)
        success_rate = results.get("success_rate", 0.0)
        
        print(f"Total files: {total_files}")
        print(f"Successful: {successful_files}")
        print(f"Failed: {failed_files}")
        print(f"Success rate: {success_rate:.1f}%")
        
        file_results = results.get("file_results", {})
        if file_results:
            print(f"\nDetailed results:")
            
            for file, file_result in file_results.items():
                if file_result.get("success", False):
                    validation_result = file_result.get("validation_result", {})
                    instance_id = file_result.get("instance_id", "unknown")
                    status = validation_result.get("status", "unknown")
                    
                    if status == "success":
                        print(f"  {file} ({instance_id}): All tests passed")
                    elif status == "test_mismatch":
                        print(f"  {file} ({instance_id}): Some tests failed")
                    elif status == "report_not_found":
                        print(f"  {file} ({instance_id}): Report not found")
                    elif status == "read_error":
                        print(f"  {file} ({instance_id}): Error reading report")
                    else:
                        print(f"  {file} ({instance_id}): Unknown status")
                else:
                    error = file_result.get("error", "unknown")
                    print(f"  {file}: {error}")
        
        if success_rate == 100.0:
            print(f"\n All files processed successfully!")
        elif success_rate >= 80.0:
            print(f"\n Most files processed successfully")
        elif success_rate >= 50.0:
            print(f"\n Half of files processed")
        else:
            print(f"\n Most files failed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SWE-bench data points validation")
    parser.add_argument("--data-dir", default="data_points", 
                       help="Directory with data point JSON files")
    parser.add_argument("--files", nargs="+", 
                       help="Specific file names for validation (without .json extension)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = Validator(data_dir=args.data_dir)
    results = validator.validate(args.files)
    validator.display_results(results)
    
    if "error" in results:
        sys.exit(1)
    elif results.get("success_rate", 0) == 100.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()