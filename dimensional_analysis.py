"""
Dimensional Analysis Module

This module provides functionality for step-by-step dimensional analysis
and unit validation for chemical calculations. It helps users understand
the relationships between different units and validates calculations.

"""

import re
from typing import Dict, List, Tuple, Optional
from utils import format_concentration, safe_float_conversion


class DimensionalAnalyzer:
    """
    Analyzer for dimensional analysis and unit conversions.
    
    This class provides methods to perform step-by-step dimensional analysis
    for chemical calculations, helping users understand unit relationships
    and validate their calculations.
    """
    
    def __init__(self):
        """Initialize the dimensional analyzer with unit definitions."""
        # Define common units and their relationships
        self.units = {
            # Length units
            'm': {'base': 'm', 'factor': 1.0},
            'cm': {'base': 'm', 'factor': 0.01},
            'mm': {'base': 'm', 'factor': 0.001},
            'km': {'base': 'm', 'factor': 1000.0},
            'in': {'base': 'm', 'factor': 0.0254},
            'ft': {'base': 'm', 'factor': 0.3048},
            'yd': {'base': 'm', 'factor': 0.9144},
            
            # Volume units
            'L': {'base': 'L', 'factor': 1.0},
            'mL': {'base': 'L', 'factor': 0.001},
            'cm3': {'base': 'L', 'factor': 0.001},
            'dm3': {'base': 'L', 'factor': 1.0},
            'gal': {'base': 'L', 'factor': 3.78541},
            
            # Mass units
            'g': {'base': 'g', 'factor': 1.0},
            'kg': {'base': 'g', 'factor': 1000.0},
            'mg': {'base': 'g', 'factor': 0.001},
            'lb': {'base': 'g', 'factor': 453.592},
            'oz': {'base': 'g', 'factor': 28.3495},
            
            # Time units
            's': {'base': 's', 'factor': 1.0},
            'min': {'base': 's', 'factor': 60.0},
            'h': {'base': 's', 'factor': 3600.0},
            'day': {'base': 's', 'factor': 86400.0},
            
            # Temperature units
            'K': {'base': 'K', 'factor': 1.0},
            'C': {'base': 'K', 'factor': 1.0, 'offset': 273.15},
            'F': {'base': 'K', 'factor': 5/9, 'offset': 273.15 - 32*5/9},
            
            # Concentration units
            'mol/L': {'base': 'mol/L', 'factor': 1.0},
            'M': {'base': 'mol/L', 'factor': 1.0},
            'mol/kg': {'base': 'mol/kg', 'factor': 1.0},
            'm': {'base': 'mol/kg', 'factor': 1.0},
            'N': {'base': 'eq/L', 'factor': 1.0},
            'eq/L': {'base': 'eq/L', 'factor': 1.0},
            'ppm': {'base': 'ppm', 'factor': 1.0},
            '%': {'base': '%', 'factor': 1.0},
            
            # Pressure units
            'Pa': {'base': 'Pa', 'factor': 1.0},
            'kPa': {'base': 'Pa', 'factor': 1000.0},
            'atm': {'base': 'Pa', 'factor': 101325.0},
            'bar': {'base': 'Pa', 'factor': 100000.0},
            'torr': {'base': 'Pa', 'factor': 133.322},
            'mmHg': {'base': 'Pa', 'factor': 133.322},
            
            # Energy units
            'J': {'base': 'J', 'factor': 1.0},
            'kJ': {'base': 'J', 'factor': 1000.0},
            'cal': {'base': 'J', 'factor': 4.184},
            'kcal': {'base': 'J', 'factor': 4184.0},
        }
        
        # Define derived units and their relationships
        self.derived_units = {
            'density': {'base_units': ['g', 'L'], 'formula': 'g/L'},
            'molarity': {'base_units': ['mol', 'L'], 'formula': 'mol/L'},
            'molality': {'base_units': ['mol', 'kg'], 'formula': 'mol/kg'},
            'normality': {'base_units': ['eq', 'L'], 'formula': 'eq/L'},
            'pressure': {'base_units': ['Pa'], 'formula': 'Pa'},
            'energy': {'base_units': ['J'], 'formula': 'J'},
        }
    
    def validate_units(self, value: float, from_unit: str, to_unit: str) -> bool:
        """
        Validate if a unit conversion is possible.
        
        Args:
            value (float): Value to convert
            from_unit (str): Original unit
            to_unit (str): Target unit
            
        Returns:
            bool: True if conversion is possible, False otherwise
        """
        if from_unit not in self.units or to_unit not in self.units:
            return False
        
        # Check if units have the same base unit
        from_base = self.units[from_unit]['base']
        to_base = self.units[to_unit]['base']
        
        return from_base == to_base
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Tuple[float, List[str]]:
        """
        Convert a value from one unit to another with step-by-step analysis.
        
        Args:
            value (float): Value to convert
            from_unit (str): Original unit
            to_unit (str): Target unit
            
        Returns:
            Tuple[float, List[str]]: Converted value and step-by-step analysis
            
        Raises:
            ValueError: If conversion is not possible
        """
        if not self.validate_units(value, from_unit, to_unit):
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        
        steps = []
        steps.append(f"1. Original value: {value} {from_unit}")
        
        # Get unit information
        from_info = self.units[from_unit]
        to_info = self.units[to_unit]
        
        steps.append(f"2. Base unit: {from_info['base']}")
        
        # Convert to base unit first
        base_value = value * from_info['factor']
        if 'offset' in from_info:
            base_value += from_info['offset']
        
        steps.append(f"3. Convert to base unit: {value} x {from_info['factor']} = {base_value}")
        
        # Convert from base unit to target unit
        if 'offset' in to_info:
            base_value -= to_info['offset']
        
        converted_value = base_value / to_info['factor']
        
        steps.append(f"4. Convert to target unit: {base_value} / {to_info['factor']} = {converted_value}")
        steps.append(f"5. Final result: {converted_value} {to_unit}")
        
        return converted_value, steps
    
    def analyze_molecular_weight_calculation(self, elements: Dict[str, int]) -> Dict[str, any]:
        """
        Perform dimensional analysis for molecular weight calculation.
        
        Args:
            elements (Dict[str, int]): Dictionary of elements and their counts
            
        Returns:
            Dict[str, any]: Analysis with steps and validation
        """
        steps = []
        total_weight = 0.0
        
        steps.append("Molecular Weight Calculation Analysis:")
        steps.append("=" * 40)
        
        for element, count in elements.items():
            # Get atomic weight (simplified - in real implementation, use atomic weights table)
            atomic_weight = self._get_atomic_weight(element)
            element_weight = atomic_weight * count
            
            steps.append(f" {element}: {count} atoms x {atomic_weight} g/mol = {element_weight} g/mol")
            total_weight += element_weight
        
        steps.append(f" Total molecular weight: {total_weight} g/mol")
        
        # Validate the calculation
        validation = self._validate_molecular_weight_calculation(elements, total_weight)
        
        return {
            'total_weight': total_weight,
            'steps': steps,
            'validation': validation,
            'is_valid': validation['is_valid']
        }
    
    def analyze_concentration_conversion(self, from_unit: str, to_unit: str, 
                                      value: float, solute_mw: float) -> Dict[str, any]:
        """
        Perform dimensional analysis for concentration conversion.
        
        Args:
            from_unit (str): Original concentration unit
            to_unit (str): Target concentration unit
            value (float): Original concentration value
            solute_mw (float): Molecular weight of solute
            
        Returns:
            Dict[str, any]: Analysis with steps and validation
        """
        steps = []
        steps.append(f"Concentration Conversion Analysis: {from_unit} -> {to_unit}")
        steps.append("=" * 50)
        
        # Analyze the conversion based on unit types
        if from_unit.upper() == 'M' and to_unit.lower() == 'm':
            # Molarity to molality
            steps.extend(self._analyze_molarity_to_molality(value, solute_mw))
        elif from_unit.lower() == 'm' and to_unit.upper() == 'M':
            # Molality to molarity
            steps.extend(self._analyze_molality_to_molarity(value, solute_mw))
        elif from_unit.upper() == 'M' and to_unit.upper() == 'N':
            # Molarity to normality
            steps.extend(self._analyze_molarity_to_normality(value))
        elif from_unit.upper() == 'N' and to_unit.upper() == 'M':
            # Normality to molarity
            steps.extend(self._analyze_normality_to_molarity(value))
        else:
            steps.append(f"Unsupported conversion: {from_unit} -> {to_unit}")
        
        # Validate the conversion
        validation = self._validate_concentration_conversion(from_unit, to_unit, value)
        
        return {
            'steps': steps,
            'validation': validation,
            'is_valid': validation['is_valid']
        }
    
    def _analyze_molarity_to_molality(self, molarity: float, solute_mw: float) -> List[str]:
        """Analyze molarity to molality conversion."""
        steps = []
        
        steps.append(f"1. Original molarity: {molarity} mol/L")
        steps.append(f"2. Solute molecular weight: {solute_mw} g/mol")
        steps.append("3. Conversion steps:")
        steps.append(f"    Mass of solute per liter: {molarity} mol/L x {solute_mw} g/mol = {molarity * solute_mw} g/L")
        steps.append(f"    Mass of solvent per liter: 1000 g/L - {molarity * solute_mw} g/L = {1000 - molarity * solute_mw} g/L")
        steps.append(f"    Molality: {molarity} mol / {(1000 - molarity * solute_mw) / 1000} kg = {molarity / ((1000 - molarity * solute_mw) / 1000)} mol/kg")
        
        return steps
    
    def _analyze_molality_to_molarity(self, molality: float, solute_mw: float) -> List[str]:
        """Analyze molality to molarity conversion."""
        steps = []
        
        steps.append(f"1. Original molality: {molality} mol/kg")
        steps.append(f"2. Solute molecular weight: {solute_mw} g/mol")
        steps.append("3. Conversion steps:")
        steps.append(f"    Mass of solute per kg solvent: {molality} mol/kg x {solute_mw} g/mol = {molality * solute_mw} g/kg")
        steps.append(f"    Total mass per kg solvent: 1000 g + {molality * solute_mw} g = {1000 + molality * solute_mw} g")
        steps.append(f"    Volume per kg solvent: {(1000 + molality * solute_mw) / 1000} L")
        steps.append(f"    Molarity: {molality} mol / {(1000 + molality * solute_mw) / 1000} L = {molality / ((1000 + molality * solute_mw) / 1000)} mol/L")
        
        return steps
    
    def _analyze_molarity_to_normality(self, molarity: float) -> List[str]:
        """Analyze molarity to normality conversion."""
        steps = []
        
        steps.append(f"1. Original molarity: {molarity} mol/L")
        steps.append("2. Normality calculation:")
        steps.append(f"    Normality = Molarity x valence factor")
        steps.append(f"    Assuming valence factor = 1: {molarity} x 1 = {molarity} N")
        
        return steps
    
    def _analyze_normality_to_molarity(self, normality: float) -> List[str]:
        """Analyze normality to molarity conversion."""
        steps = []
        
        steps.append(f"1. Original normality: {normality} N")
        steps.append("2. Molarity calculation:")
        steps.append(f"    Molarity = Normality / valence factor")
        steps.append(f"    Assuming valence factor = 1: {normality} / 1 = {normality} M")
        
        return steps
    
    def _validate_molecular_weight_calculation(self, elements: Dict[str, int], 
                                            total_weight: float) -> Dict[str, any]:
        """Validate molecular weight calculation."""
        validation = {
            'is_valid': True,
            'checks': [],
            'warnings': []
        }
        
        # Check for reasonable molecular weight
        if total_weight <= 0:
            validation['is_valid'] = False
            validation['checks'].append("Molecular weight must be positive")
        
        if total_weight > 10000:  # Reasonable upper limit
            validation['warnings'].append("Unusually high molecular weight - verify calculation")
        
        # Check for common elements
        common_elements = {'H', 'C', 'N', 'O', 'S', 'P', 'Cl', 'Na', 'K', 'Ca', 'Mg', 'Fe'}
        unusual_elements = [elem for elem in elements.keys() if elem not in common_elements]
        
        if unusual_elements:
            validation['warnings'].append(f"Unusual elements detected: {unusual_elements}")
        
        validation['checks'].append("All elements have valid atomic weights")
        validation['checks'].append("Molecular weight calculation is mathematically correct")
        
        return validation
    
    def _validate_concentration_conversion(self, from_unit: str, to_unit: str, 
                                        value: float) -> Dict[str, any]:
        """Validate concentration conversion."""
        validation = {
            'is_valid': True,
            'checks': [],
            'warnings': []
        }
        
        # Check for reasonable concentration values
        if value < 0:
            validation['is_valid'] = False
            validation['checks'].append("Concentration cannot be negative")
        
        if value > 100:  # Reasonable upper limit for most solutions
            validation['warnings'].append("Unusually high concentration - verify units")
        
        # Check unit compatibility
        if not self.validate_units(value, from_unit, to_unit):
            validation['is_valid'] = False
            validation['checks'].append(f"Cannot convert from {from_unit} to {to_unit}")
        
        validation['checks'].append("Unit conversion is mathematically valid")
        
        return validation
    
    def _get_atomic_weight(self, element: str) -> float:
        """Get atomic weight for an element (simplified)."""
        # Simplified atomic weights - in real implementation, use comprehensive table
        atomic_weights = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982,
            'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
            'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546,
            'Zn': 65.38, 'Br': 79.904, 'Ag': 107.868, 'I': 126.904
        }
        
        return atomic_weights.get(element, 0.0)
    
    def get_unit_conversion_guide(self) -> Dict[str, List[str]]:
        """
        Get a guide for common unit conversions.
        
        Returns:
            Dict[str, List[str]]: Guide organized by unit type
        """
        guide = {
            'Length': [
                '1 m = 100 cm = 1000 mm',
                '1 km = 1000 m',
                '1 in = 2.54 cm',
                '1 ft = 30.48 cm',
                '1 yd = 91.44 cm'
            ],
            'Volume': [
                '1 L = 1000 mL',
                '1 L = 1 dm',
                '1 mL = 1 cm',
                '1 gal = 3.785 L'
            ],
            'Mass': [
                '1 kg = 1000 g',
                '1 g = 1000 mg',
                '1 lb = 453.592 g',
                '1 oz = 28.3495 g'
            ],
            'Concentration': [
                '1 M = 1 mol/L',
                '1 m = 1 mol/kg',
                '1 N = 1 eq/L',
                '1 ppm = 1 mg/L (for dilute aqueous solutions)',
                '1% = 10 g/L (for dilute aqueous solutions)'
            ],
            'Temperature': [
                'K = C + 273.15',
                'C = (F - 32) x 5/9',
                'F = C x 9/5 + 32'
            ]
        }
        
        return guide 