"""
Data loading utilities with automatic task type detection.

Supports:
- Multi-class classification (single integer label per sample)
- Multi-label classification (array of labels per sample)
- Multiple task groups (e.g., CN has NES and NP with different task types)
"""

import ast
import re
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from dataclasses import dataclass


@dataclass
class TaskInfo:
    """Information about a classification task in the dataset."""
    name: str
    task_type: str  # "class" for multi-class, "label" for multi-label
    columns: List[str]  # all columns belonging to this task (including annotator columns)
    ground_truth_column: Optional[str]  # the consensus/ground truth column if exists


@dataclass
class DatasetInfo:
    """Complete information about a loaded dataset."""
    df: pd.DataFrame
    text_column: str
    tasks: Dict[str, TaskInfo]  # task_name -> TaskInfo
    

def _is_array_like(value: Any) -> bool:
    """Check if a value represents an array/list (either actual list or string representation)."""
    if isinstance(value, (list, tuple)):
        return True
    if isinstance(value, str):
        # Check for list-like string patterns: "[1, 2]", "[3]", etc.
        stripped = value.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                parsed = ast.literal_eval(stripped)
                return isinstance(parsed, list)
            except (ValueError, SyntaxError):
                pass
    return False


def _is_integer_like(value: Any) -> bool:
    """Check if a value is integer-like (int, float that's whole, or numeric string)."""
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return value.is_integer()
        return True
    if isinstance(value, str):
        try:
            float_val = float(value)
            return float_val.is_integer()
        except ValueError:
            return False
    return False


def _is_categorical_string(value: Any) -> bool:
    """Check if a value is a categorical string label (not array-like, not numeric)."""
    if pd.isna(value):
        return False
    if not isinstance(value, str):
        return False
    # Not array-like
    if _is_array_like(value):
        return False
    # Not numeric
    try:
        float(value)
        return False
    except ValueError:
        pass
    # It's a non-numeric string
    return True


def _detect_column_task_type(series: pd.Series) -> str:
    """
    Detect whether a column represents multi-class or multi-label classification.
    
    Returns:
        "class" for multi-class (single integer values or categorical strings)
        "label" for multi-label (array/list values)
        "unknown" if unable to determine
    """
    # Get non-null values for detection
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown"
    
    # Sample up to 10 values for detection
    sample_size = min(10, len(non_null))
    sample_values = non_null.head(sample_size).tolist()
    
    # Check if values are array-like (multi-label)
    array_count = sum(1 for v in sample_values if _is_array_like(v))
    if array_count > sample_size / 2:
        return "label"
    
    # Check if values are integer-like (multi-class numeric)
    int_count = sum(1 for v in sample_values if _is_integer_like(v))
    if int_count > sample_size / 2:
        return "class"
    
    # Check if values are categorical strings (multi-class categorical)
    cat_count = sum(1 for v in sample_values if _is_categorical_string(v))
    if cat_count > sample_size / 2:
        return "class"
    
    return "unknown"


def _extract_prefix(column_name: str) -> Tuple[str, Optional[str]]:
    """
    Extract the task prefix from a column name.
    
    Examples:
        "NES(A1)" -> ("NES", "A1")
        "NES" -> ("NES", None)
        "Label" -> ("Label", None)
        "ES(A1)" -> ("ES", "A1")
    
    Returns:
        Tuple of (prefix, annotator_id or None)
    """
    # Pattern for columns with annotator suffix: PREFIX(A1), PREFIX(A2), etc.
    match = re.match(r'^([A-Za-z_]+)\(([A-Za-z0-9]+)\)$', column_name)
    if match:
        return match.group(1), match.group(2)
    
    # No annotator suffix
    return column_name, None


def _group_columns_by_prefix(columns: List[str]) -> Dict[str, List[str]]:
    """
    Group columns by their task prefix.
    
    Returns:
        Dict mapping prefix to list of column names
    """
    groups: Dict[str, List[str]] = {}
    
    for col in columns:
        prefix, _ = _extract_prefix(col)
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(col)
    
    return groups


def _find_ground_truth_column(prefix: str, columns: List[str]) -> Optional[str]:
    """
    Find the ground truth column for a task prefix.
    
    Ground truth column is typically the one without annotator suffix,
    or a column named exactly as the prefix.
    """
    for col in columns:
        col_prefix, annotator = _extract_prefix(col)
        if col_prefix == prefix and annotator is None:
            return col
    return None


def detect_task_types(df: pd.DataFrame, text_column: str = "Text") -> Dict[str, TaskInfo]:
    """
    Automatically detect task types from a DataFrame.
    
    Groups columns by prefix and determines task type for each group.
    
    Args:
        df: The dataset DataFrame
        text_column: Name of the text column (excluded from task detection)
    
    Returns:
        Dict mapping task name to TaskInfo
    """
    # Get all columns except text
    task_columns = [col for col in df.columns if col != text_column]
    
    # Group columns by prefix
    prefix_groups = _group_columns_by_prefix(task_columns)
    
    tasks: Dict[str, TaskInfo] = {}
    
    for prefix, columns in prefix_groups.items():
        # Find the ground truth column (if it exists)
        gt_column = _find_ground_truth_column(prefix, columns)
        
        # Use ground truth column for type detection if available,
        # otherwise use the first annotator column
        detection_column = gt_column if gt_column else columns[0]
        task_type = _detect_column_task_type(df[detection_column])
        
        tasks[prefix] = TaskInfo(
            name=prefix,
            task_type=task_type,
            columns=columns,
            ground_truth_column=gt_column
        )
    
    return tasks


def load_dataset(
    data_path: str,
    task_type_override: Optional[str] = None,
    text_column: str = "Text"
) -> DatasetInfo:
    """
    Load a dataset and detect task types.
    
    Args:
        data_path: Path to the Excel file
        task_type_override: If provided ("class" or "label"), override auto-detection
                           for all tasks
        text_column: Name of the text column
    
    Returns:
        DatasetInfo with the loaded data and task information
    """
    df = pd.read_excel(data_path)
    
    # Detect task types
    tasks = detect_task_types(df, text_column)
    
    # Apply override if specified
    if task_type_override is not None:
        if task_type_override not in ("class", "label"):
            raise ValueError(f"task_type must be 'class' or 'label', got '{task_type_override}'")
        for task in tasks.values():
            task.task_type = task_type_override
    
    return DatasetInfo(
        df=df,
        text_column=text_column,
        tasks=tasks
    )


def print_dataset_info(info: DatasetInfo) -> str:
    """Generate a summary string of the dataset info."""
    lines = [
        f"Dataset loaded: {len(info.df)} samples",
        f"Text column: {info.text_column}",
        f"Tasks detected: {len(info.tasks)}",
        ""
    ]
    
    for task_name, task_info in info.tasks.items():
        lines.append(f"  Task: {task_name}")
        lines.append(f"    Type: {task_info.task_type}")
        lines.append(f"    Columns: {task_info.columns}")
        if task_info.ground_truth_column:
            lines.append(f"    Ground Truth: {task_info.ground_truth_column}")
        lines.append("")
    
    return "\n".join(lines)

