"""
Concentration Converter Module

This module provides functionality to convert between different concentration units
including molarity, molality, and normality. It includes dimensional analysis
validation for all conversions.

"""

import math
from typing import Dict, List, Tuple, Optional
from utils import format_concentration, safe_float_conversion


class ConcentrationConverter:
    """
    Converter for concentration units with dimensional analysis validation.
    
    This class provides methods to convert between different concentration units
    including molarity (M), molality (m), and normality (N). All conversions
    include step-by-step dimensional analysis for educational purposes.
    """
    
    def __init__(self):
        """Initialize the concentration converter."""
        # Standard density of water at 25C (g/mL)
        self.water_density = 0.997
        
        # Common solvent densities (g/mL) at 25C
        self.solvent_densities = {
            'water': 0.997,
            'ethanol': 0.789,
            'methanol': 0.791,
            'acetone': 0.784,
            'toluene': 0.867,
            'hexane': 0.659,
            'dichloromethane': 1.33,
            'chloroform': 1.49
        }
    
    def molarity_to_molality(self, molarity: float, solute_mw: float, 
                           solvent_density: float = None) -> float:
        """
        Convert molarity (M) to molality (m).
        
        Formula: m = M / (density - M * MW_solute / 1000)
        
        Args:
            molarity (float): Concentration in mol/L
            solute_mw (float): Molecular weight of solute in g/mol
            solvent_density (float): Density of solvent in g/mL (default: water)
            
        Returns:
            float: Concentration in mol/kg
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if molarity < 0:
            raise ValueError("Molarity cannot be negative")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if solvent_density is None:
            solvent_density = self.water_density
        
        if solvent_density <= 0:
            raise ValueError("Solvent density must be positive")
        
        # Convert density from g/mL to g/L
        density_g_l = solvent_density * 1000
        
        # Calculate molality using the formula
        # m = M / (density - M * MW_solute / 1000)
        denominator = density_g_l - (molarity * solute_mw / 1000)
        
        if denominator <= 0:
            raise ValueError("Invalid concentration: solution density too low")
        
        molality = molarity / denominator
        
        return molality
    
    def molality_to_molarity(self, molality: float, solute_mw: float, 
                           solvent_density: float = None) -> float:
        """
        Convert molality (m) to molarity (M).
        
        Formula: M = m * density / (1 + m * MW_solute / 1000)
        
        Args:
            molality (float): Concentration in mol/kg
            solute_mw (float): Molecular weight of solute in g/mol
            solvent_density (float): Density of solvent in g/mL (default: water)
            
        Returns:
            float: Concentration in mol/L
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if molality < 0:
            raise ValueError("Molality cannot be negative")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if solvent_density is None:
            solvent_density = self.water_density
        
        if solvent_density <= 0:
            raise ValueError("Solvent density must be positive")
        
        # Convert density from g/mL to g/L
        density_g_l = solvent_density * 1000
        
        # Calculate molarity using the formula
        # M = m * density / (1 + m * MW_solute / 1000)
        denominator = 1 + (molality * solute_mw / 1000)
        
        if denominator <= 0:
            raise ValueError("Invalid concentration")
        
        molarity = (molality * density_g_l) / denominator
        
        return molarity
    
    def molarity_to_normality(self, molarity: float, solute_mw: float, 
                            valence_factor: int = 1) -> float:
        """
        Convert molarity (M) to normality (N).
        
        Formula: N = M * valence_factor
        
        Args:
            molarity (float): Concentration in mol/L
            solute_mw (float): Molecular weight of solute in g/mol
            valence_factor (int): Number of equivalents per mole (default: 1)
            
        Returns:
            float: Concentration in equivalents/L (N)
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if molarity < 0:
            raise ValueError("Molarity cannot be negative")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if valence_factor <= 0:
            raise ValueError("Valence factor must be positive")
        
        normality = molarity * valence_factor
        
        return normality
    
    def normality_to_molarity(self, normality: float, solute_mw: float, 
                            valence_factor: int = 1) -> float:
        """
        Convert normality (N) to molarity (M).
        
        Formula: M = N / valence_factor
        
        Args:
            normality (float): Concentration in equivalents/L
            solute_mw (float): Molecular weight of solute in g/mol
            valence_factor (int): Number of equivalents per mole (default: 1)
            
        Returns:
            float: Concentration in mol/L
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if normality < 0:
            raise ValueError("Normality cannot be negative")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if valence_factor <= 0:
            raise ValueError("Valence factor must be positive")
        
        molarity = normality / valence_factor
        
        return molarity
    
    def calculate_mass_percent(self, molarity: float, solute_mw: float, 
                             solvent_density: float = None) -> float:
        """
        Calculate mass percent from molarity.
        
        Formula: mass% = (M * MW_solute * 100) / (density * 1000)
        
        Args:
            molarity (float): Concentration in mol/L
            solute_mw (float): Molecular weight of solute in g/mol
            solvent_density (float): Density of solvent in g/mL (default: water)
            
        Returns:
            float: Mass percent (%)
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if molarity < 0:
            raise ValueError("Molarity cannot be negative")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if solvent_density is None:
            solvent_density = self.water_density
        
        if solvent_density <= 0:
            raise ValueError("Solvent density must be positive")
        
        # Convert density from g/mL to g/L
        density_g_l = solvent_density * 1000
        
        mass_percent = (molarity * solute_mw * 100) / density_g_l
        
        return mass_percent
    
    def calculate_molarity_from_mass_percent(self, mass_percent: float, solute_mw: float,
                                           solvent_density: float = None) -> float:
        """
        Calculate molarity from mass percent.
        
        Formula: M = (mass% * density * 10) / MW_solute
        
        Args:
            mass_percent (float): Mass percent (%)
            solute_mw (float): Molecular weight of solute in g/mol
            solvent_density (float): Density of solvent in g/mL (default: water)
            
        Returns:
            float: Concentration in mol/L
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if mass_percent < 0 or mass_percent > 100:
            raise ValueError("Mass percent must be between 0 and 100")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if solvent_density is None:
            solvent_density = self.water_density
        
        if solvent_density <= 0:
            raise ValueError("Solvent density must be positive")
        
        molarity = (mass_percent * solvent_density * 10) / solute_mw
        
        return molarity
    
    def calculate_parts_per_million(self, molarity: float, solute_mw: float,
                                  solvent_density: float = None) -> float:
        """
        Calculate parts per million (ppm) from molarity.
        
        Formula: ppm = (M * MW_solute * 1000000) / (density * 1000)
        
        Args:
            molarity (float): Concentration in mol/L
            solute_mw (float): Molecular weight of solute in g/mol
            solvent_density (float): Density of solvent in g/mL (default: water)
            
        Returns:
            float: Concentration in ppm
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if molarity < 0:
            raise ValueError("Molarity cannot be negative")
        
        if solute_mw <= 0:
            raise ValueError("Molecular weight must be positive")
        
        if solvent_density is None:
            solvent_density = self.water_density
        
        if solvent_density <= 0:
            raise ValueError("Solvent density must be positive")
        
        # Convert density from g/mL to g/L
        density_g_l = solvent_density * 1000
        
        ppm = (molarity * solute_mw * 1000000) / density_g_l
        
        return ppm
    
    def get_conversion_analysis(self, from_unit: str, to_unit: str, value: float,
                              solute_mw: float, **kwargs) -> Dict[str, any]:
        """
        Get a comprehensive analysis of a concentration conversion.
        
        Args:
            from_unit (str): Original unit (M, m, N, %, ppm)
            to_unit (str): Target unit (M, m, N, %, ppm)
            value (float): Original concentration value
            solute_mw (float): Molecular weight of solute in g/mol
            **kwargs: Additional parameters (solvent_density, valence_factor, etc.)
            
        Returns:
            Dict[str, any]: Comprehensive conversion analysis
        """
        try:
            # Validate input parameters
            if value < 0:
                raise ValueError("Concentration value cannot be negative")
            
            if solute_mw <= 0:
                raise ValueError("Molecular weight must be positive")
            
            # Get solvent density
            solvent_density = kwargs.get('solvent_density', self.water_density)
            valence_factor = kwargs.get('valence_factor', 1)
            
            # Perform conversion based on unit types
            converted_value = None
            conversion_steps = []
            
            if from_unit.upper() == 'M' and to_unit.lower() == 'm':
                converted_value = self.molarity_to_molality(value, solute_mw, solvent_density)
                conversion_steps = [
                    f"1. Original molarity: {value} mol/L",
                    f"2. Solute molecular weight: {solute_mw} g/mol",
                    f"3. Solvent density: {solvent_density} g/mL",
                    f"4. Converted molality: {converted_value} mol/kg"
                ]
            
            elif from_unit.lower() == 'm' and to_unit.upper() == 'M':
                converted_value = self.molality_to_molarity(value, solute_mw, solvent_density)
                conversion_steps = [
                    f"1. Original molality: {value} mol/kg",
                    f"2. Solute molecular weight: {solute_mw} g/mol",
                    f"3. Solvent density: {solvent_density} g/mL",
                    f"4. Converted molarity: {converted_value} mol/L"
                ]
            
            elif from_unit.upper() == 'M' and to_unit.upper() == 'N':
                converted_value = self.molarity_to_normality(value, solute_mw, valence_factor)
                conversion_steps = [
                    f"1. Original molarity: {value} mol/L",
                    f"2. Valence factor: {valence_factor}",
                    f"3. Converted normality: {converted_value} N"
                ]
            
            elif from_unit.upper() == 'N' and to_unit.upper() == 'M':
                converted_value = self.normality_to_molarity(value, solute_mw, valence_factor)
                conversion_steps = [
                    f"1. Original normality: {value} N",
                    f"2. Valence factor: {valence_factor}",
                    f"3. Converted molarity: {converted_value} mol/L"
                ]
            
            elif from_unit.upper() == 'M' and to_unit == '%':
                converted_value = self.calculate_mass_percent(value, solute_mw, solvent_density)
                conversion_steps = [
                    f"1. Original molarity: {value} mol/L",
                    f"2. Solute molecular weight: {solute_mw} g/mol",
                    f"3. Solvent density: {solvent_density} g/mL",
                    f"4. Converted mass percent: {converted_value}%"
                ]
            
            elif from_unit == '%' and to_unit.upper() == 'M':
                converted_value = self.calculate_molarity_from_mass_percent(value, solute_mw, solvent_density)
                conversion_steps = [
                    f"1. Original mass percent: {value}%",
                    f"2. Solute molecular weight: {solute_mw} g/mol",
                    f"3. Solvent density: {solvent_density} g/mL",
                    f"4. Converted molarity: {converted_value} mol/L"
                ]
            
            elif from_unit.upper() == 'M' and to_unit.lower() == 'ppm':
                converted_value = self.calculate_parts_per_million(value, solute_mw, solvent_density)
                conversion_steps = [
                    f"1. Original molarity: {value} mol/L",
                    f"2. Solute molecular weight: {solute_mw} g/mol",
                    f"3. Solvent density: {solvent_density} g/mL",
                    f"4. Converted ppm: {converted_value} ppm"
                ]
            
            else:
                raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")
            
            analysis = {
                'from_unit': from_unit,
                'to_unit': to_unit,
                'original_value': value,
                'converted_value': converted_value,
                'solute_mw': solute_mw,
                'solvent_density': solvent_density,
                'conversion_steps': conversion_steps,
                'is_valid': True,
                'error': None
            }
            
            return analysis
            
        except ValueError as e:
            return {
                'from_unit': from_unit,
                'to_unit': to_unit,
                'original_value': value,
                'converted_value': None,
                'solute_mw': solute_mw,
                'solvent_density': kwargs.get('solvent_density', self.water_density),
                'conversion_steps': [],
                'is_valid': False,
                'error': str(e)
            }
    
    def get_solvent_density(self, solvent_name: str) -> float:
        """
        Get the density of a common solvent.
        
        Args:
            solvent_name (str): Name of the solvent
            
        Returns:
            float: Density in g/mL
            
        Raises:
            ValueError: If solvent not found
        """
        solvent_name_lower = solvent_name.lower()
        
        if solvent_name_lower not in self.solvent_densities:
            raise ValueError(f"Unknown solvent: {solvent_name}")
        
        return self.solvent_densities[solvent_name_lower]
    
    def list_available_solvents(self) -> List[str]:
        """
        Get a list of available solvents with their densities.
        
        Returns:
            List[str]: List of solvent names
        """
        return list(self.solvent_densities.keys()) 