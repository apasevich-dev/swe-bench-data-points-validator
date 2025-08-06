"""
Formatter for converting data points to SWE-bench prediction format.
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PredictionConverter:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
    
    def convert(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        logger.info(f"Converting {len(data_points)} data points to predictions format")
        
        predictions = []
        converted_count = 0
        failed_count = 0
        
        for data_point in data_points:
            prediction = self._convert_single(data_point)
            
            if prediction:
                predictions.append(prediction)
                converted_count += 1
            else:
                failed_count += 1
                logger.warning(f"Failed to convert data point: {data_point.get('instance_id', 'unknown')}")
        
        logger.info(f"Conversion completed: {converted_count} successful, {failed_count} failed")
        return predictions
    
    def _convert_single(self, data_point: Dict[str, Any]) -> Optional[Dict[str, str]]:
        instance_id = data_point.get("instance_id")
        patch = data_point.get("patch")
        
        if not instance_id or not patch:
            logger.warning(f"Missing required fields in data point: instance_id={instance_id}, patch={'present' if patch else 'missing'}")
            return None
        
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": self.model_name,
            "model_patch": patch
        }
        
        return prediction
    
    def save_to_file(self, predictions: List[Dict[str, str]], file_path: str) -> bool:
        logger.info(f"Saving {len(predictions)} predictions to file: {file_path}")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for prediction in predictions:
                    f.write(json.dumps(prediction) + '\n')
            
            logger.info(f"Successfully saved predictions to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save predictions to {file_path}: {e}")
            return False