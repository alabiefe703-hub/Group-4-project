"""
Utility functions for the Chemical Analysis CLI Tool.

This module provides helper functions used throughout the application
including file validation, screen clearing, and data formatting utilities.


"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional


def clear_screen():
    """
    Clear the terminal screen in a cross-platform manner.
    
    Uses different commands for Windows vs Unix-like systems.
    """
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix-like systems (Linux, macOS)
        os.system('clear')


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is readable.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        bool: True if file exists and is readable, False otherwise
    """
    if not file_path:
        return False
    
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def format_chemical_formula(formula: str) -> str:
    """
    Format a chemical formula for consistent display.
    
    Args:
        formula (str): Raw chemical formula
        
    Returns:
        str: Formatted chemical formula
    """
    if not formula:
        return ""
    
    # Remove extra whitespace and capitalize
    formatted = formula.strip().upper()
    
    # Ensure proper spacing around operators
    formatted = re.sub(r'\s*\+\s*', ' + ', formatted)
    formatted = re.sub(r'\s*->\s*', ' -> ', formatted)
    
    return formatted


def parse_number_with_units(value: str) -> Tuple[float, str]:
    """
    Parse a number with optional units from a string.
    
    Args:
        value (str): String containing number and optional units
        
    Returns:
        Tuple[float, str]: (numeric_value, unit_string)
        
    Raises:
        ValueError: If the value cannot be parsed
    """
    if not value:
        raise ValueError("Empty value provided")
    
    # Remove whitespace
    value = value.strip()
    
    # Try to extract number and units
    match = re.match(r'^([+-]?\d*\.?\d+)\s*([a-zA-Z/%]*)$', value)
    
    if not match:
        raise ValueError(f"Invalid number format: {value}")
    
    number_str, units = match.groups()
    
    try:
        number = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid number: {number_str}")
    
    return number, units.strip()


def validate_chemical_symbol(symbol: str) -> bool:
    """
    Validate if a string represents a valid chemical element symbol.
    
    Args:
        symbol (str): Element symbol to validate
        
    Returns:
        bool: True if valid element symbol, False otherwise
    """
    if not symbol:
        return False
    
    # Basic validation: first letter uppercase, second letter lowercase
    if len(symbol) == 1:
        return symbol.isupper()
    elif len(symbol) == 2:
        return symbol[0].isupper() and symbol[1].islower()
    else:
        return False


def calculate_gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor of two integers.
    
    Args:
        a (int): First integer
        b (int): Second integer
        
    Returns:
        int: Greatest common divisor
    """
    while b:
        a, b = b, a % b
    return a


def simplify_ratio(numerator: int, denominator: int) -> Tuple[int, int]:
    """
    Simplify a ratio to its lowest terms.
    
    Args:
        numerator (int): Numerator of the ratio
        denominator (int): Denominator of the ratio
        
    Returns:
        Tuple[int, int]: Simplified ratio as (numerator, denominator)
    """
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    
    gcd = calculate_gcd(abs(numerator), abs(denominator))
    return numerator // gcd, denominator // gcd


def format_molecular_weight(weight: float) -> str:
    """
    Format molecular weight with appropriate precision.
    
    Args:
        weight (float): Molecular weight value
        
    Returns:
        str: Formatted molecular weight string
    """
    if weight < 1:
        return f"{weight:.4f} g/mol"
    elif weight < 100:
        return f"{weight:.2f} g/mol"
    else:
        return f"{weight:.1f} g/mol"


def format_concentration(value: float, unit: str) -> str:
    """
    Format concentration value with appropriate precision.
    
    Args:
        value (float): Concentration value
        unit (str): Unit of concentration
        
    Returns:
        str: Formatted concentration string
    """
    if abs(value) < 0.001:
        return f"{value:.6f} {unit}"
    elif abs(value) < 1:
        return f"{value:.4f} {unit}"
    else:
        return f"{value:.2f} {unit}"


def safe_float_conversion(value: str, default: float = 0.0) -> float:
    """
    Safely convert a string to float with error handling.
    
    Args:
        value (str): String to convert
        default (float): Default value if conversion fails
        
    Returns:
        float: Converted value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def validate_equation_format(equation: str) -> bool:
    """
    Validate basic chemical equation format.
    
    Args:
        equation (str): Chemical equation to validate
        
    Returns:
        bool: True if format is valid, False otherwise
    """
    if not equation:
        return False
    
    # Check for arrow or equals sign
    if '->' not in equation and '=' not in equation:
        return False
    
    # Check for at least one reactant and one product
    parts = re.split(r'->|=', equation)
    if len(parts) != 2:
        return False
    
    reactants, products = parts
    
    # Check that both sides have content
    if not reactants.strip() or not products.strip():
        return False
    
    return True


def extract_compounds_from_equation(equation: str) -> List[str]:
    """
    Extract individual compounds from a chemical equation.
    
    Args:
        equation (str): Chemical equation
        
    Returns:
        List[str]: List of compound formulas
    """
    if not equation:
        return []
    
    # Split by arrow or equals sign
    parts = re.split(r'->|=', equation)
    if len(parts) != 2:
        return []
    
    reactants, products = parts
    
    # Extract compounds (split by + and clean up)
    compounds = []
    
    for side in [reactants, products]:
        side_compounds = [c.strip() for c in side.split('+')]
        compounds.extend([c for c in side_compounds if c])
    
    return compounds


def format_percentage(value: float) -> str:
    """
    Format a percentage value with appropriate precision.
    
    Args:
        value (float): Percentage value (0-100)
        
    Returns:
        str: Formatted percentage string
    """
    if value < 0.01:
        return f"{value:.4f}%"
    elif value < 1:
        return f"{value:.2f}%"
    else:
        return f"{value:.1f}%"


def create_sample_data() -> Dict[str, str]:
    """
    Create sample chemical formulas for testing and examples.
    
    Returns:
        Dict[str, str]: Dictionary of formula names and their formulas
    """
    return {
        "Water": "H2O",
        "Carbon Dioxide": "CO2",
        "Glucose": "C6H12O6",
        "Sulfuric Acid": "H2SO4",
        "Sodium Chloride": "NaCl",
        "Ammonia": "NH3",
        "Methane": "CH4",
        "Ethanol": "C2H5OH",
        "Nitric Acid": "HNO3",
        "Calcium Carbonate": "CaCO3"
    } 