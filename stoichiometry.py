"""
Stoichiometry Calculator Module

This module provides functionality to perform stoichiometric calculations
including limiting reactant determination, theoretical yield calculations,
and percent yield analysis.


"""

import math
from typing import Dict, List, Tuple, Optional
from parser import ChemicalFormulaParser
from molecular_calculator import MolecularCalculator


class StoichiometryCalculator:
    """
    Calculator for stoichiometric calculations and reaction analysis.
    
    This class provides methods to perform various stoichiometric calculations
    including limiting reactant analysis, theoretical yield calculations,
    and percent yield analysis.
    """
    
    def __init__(self):
        """Initialize the stoichiometry calculator with parsers."""
        self.parser = ChemicalFormulaParser()
        self.molecular_calc = MolecularCalculator()
    
    def find_limiting_reactant(self, reactants: Dict[str, float]) -> str:
        """
        Find the limiting reactant in a reaction.
        
        Args:
            reactants (Dict[str, float]): Dictionary mapping reactant formulas to their amounts (moles)
            
        Returns:
            str: Formula of the limiting reactant
            
        Raises:
            ValueError: If no reactants provided or invalid data
        """
        if not reactants:
            raise ValueError("No reactants provided")
        
        if len(reactants) < 2:
            raise ValueError("At least two reactants required for limiting reactant analysis")
        
        # For simplicity, assume 1:1 stoichiometry
        # In a real implementation, you would need the balanced equation
        limiting_reactant = min(reactants.items(), key=lambda x: x[1])
        
        return limiting_reactant[0]
    
    def calculate_theoretical_yield(self, reactant: str, reactant_moles: float, 
                                 product: str) -> float:
        """
        Calculate theoretical yield of a product.
        
        Args:
            reactant (str): Formula of the limiting reactant
            reactant_moles (float): Amount of limiting reactant in moles
            product (str): Formula of the product
            
        Returns:
            float: Theoretical yield in moles
            
        Raises:
            ValueError: If invalid data provided
        """
        if not reactant or not product:
            raise ValueError("Reactant and product formulas required")
        
        if reactant_moles <= 0:
            raise ValueError("Reactant amount must be positive")
        
        try:
            # Parse formulas to get molecular weights
            reactant_elements = self.parser.parse_formula(reactant)
            product_elements = self.parser.parse_formula(product)
            
            reactant_mw = self.molecular_calc.calculate_molecular_weight(reactant_elements)
            product_mw = self.molecular_calc.calculate_molecular_weight(product_elements)
            
            # For simplicity, assume 1:1 stoichiometry
            # In reality, you would use the balanced equation coefficients
            theoretical_moles = reactant_moles
            
            return theoretical_moles
            
        except ValueError as e:
            raise ValueError(f"Error calculating theoretical yield: {e}")
    
    def calculate_percent_yield(self, theoretical_yield: float, actual_yield: float) -> float:
        """
        Calculate percent yield of a reaction.
        
        Args:
            theoretical_yield (float): Theoretical yield in moles
            actual_yield (float): Actual yield in moles
            
        Returns:
            float: Percent yield (0-100)
            
        Raises:
            ValueError: If invalid data provided
        """
        if theoretical_yield <= 0:
            raise ValueError("Theoretical yield must be positive")
        
        if actual_yield < 0:
            raise ValueError("Actual yield cannot be negative")
        
        if actual_yield > theoretical_yield:
            raise ValueError("Actual yield cannot exceed theoretical yield")
        
        percent_yield = (actual_yield / theoretical_yield) * 100
        return percent_yield
    
    def calculate_mass_from_moles(self, moles: float, formula: str) -> float:
        """
        Calculate mass from moles using molecular weight.
        
        Args:
            moles (float): Number of moles
            formula (str): Chemical formula
            
        Returns:
            float: Mass in grams
            
        Raises:
            ValueError: If invalid data provided
        """
        if moles < 0:
            raise ValueError("Moles cannot be negative")
        
        if not formula:
            raise ValueError("Formula required")
        
        try:
            elements = self.parser.parse_formula(formula)
            molecular_weight = self.molecular_calc.calculate_molecular_weight(elements)
            
            mass = moles * molecular_weight
            return mass
            
        except ValueError as e:
            raise ValueError(f"Error calculating mass: {e}")
    
    def calculate_moles_from_mass(self, mass: float, formula: str) -> float:
        """
        Calculate moles from mass using molecular weight.
        
        Args:
            mass (float): Mass in grams
            formula (str): Chemical formula
            
        Returns:
            float: Number of moles
            
        Raises:
            ValueError: If invalid data provided
        """
        if mass < 0:
            raise ValueError("Mass cannot be negative")
        
        if not formula:
            raise ValueError("Formula required")
        
        try:
            elements = self.parser.parse_formula(formula)
            molecular_weight = self.molecular_calc.calculate_molecular_weight(elements)
            
            if molecular_weight <= 0:
                raise ValueError("Invalid molecular weight")
            
            moles = mass / molecular_weight
            return moles
            
        except ValueError as e:
            raise ValueError(f"Error calculating moles: {e}")
    
    def calculate_molar_ratio(self, reactant1: str, reactant2: str, 
                            amount1: float, amount2: float) -> float:
        """
        Calculate the molar ratio between two reactants.
        
        Args:
            reactant1 (str): Formula of first reactant
            reactant2 (str): Formula of second reactant
            amount1 (float): Amount of first reactant in moles
            amount2 (float): Amount of second reactant in moles
            
        Returns:
            float: Molar ratio (amount1/amount2)
            
        Raises:
            ValueError: If invalid data provided
        """
        if amount1 <= 0 or amount2 <= 0:
            raise ValueError("Amounts must be positive")
        
        if not reactant1 or not reactant2:
            raise ValueError("Reactant formulas required")
        
        try:
            # Validate formulas
            self.parser.parse_formula(reactant1)
            self.parser.parse_formula(reactant2)
            
            ratio = amount1 / amount2
            return ratio
            
        except ValueError as e:
            raise ValueError(f"Error calculating molar ratio: {e}")
    
    def calculate_excess_reactant(self, reactants: Dict[str, float], 
                                limiting_reactant: str) -> Dict[str, float]:
        """
        Calculate the amount of excess reactant remaining.
        
        Args:
            reactants (Dict[str, float]): Dictionary of reactants and their amounts
            limiting_reactant (str): Formula of the limiting reactant
            
        Returns:
            Dict[str, float]: Dictionary of excess reactants and their remaining amounts
            
        Raises:
            ValueError: If invalid data provided
        """
        if not reactants or limiting_reactant not in reactants:
            raise ValueError("Invalid reactants or limiting reactant")
        
        limiting_amount = reactants[limiting_reactant]
        excess_reactants = {}
        
        for reactant, amount in reactants.items():
            if reactant != limiting_reactant:
                # For simplicity, assume 1:1 stoichiometry
                # In reality, you would use the balanced equation
                excess_amount = amount - limiting_amount
                if excess_amount > 0:
                    excess_reactants[reactant] = excess_amount
        
        return excess_reactants
    
    def calculate_reaction_efficiency(self, theoretical_yield: float, 
                                   actual_yield: float) -> Dict[str, float]:
        """
        Calculate reaction efficiency metrics.
        
        Args:
            theoretical_yield (float): Theoretical yield in moles
            actual_yield (float): Actual yield in moles
            
        Returns:
            Dict[str, float]: Dictionary containing efficiency metrics
            
        Raises:
            ValueError: If invalid data provided
        """
        if theoretical_yield <= 0:
            raise ValueError("Theoretical yield must be positive")
        
        if actual_yield < 0:
            raise ValueError("Actual yield cannot be negative")
        
        percent_yield = self.calculate_percent_yield(theoretical_yield, actual_yield)
        yield_loss = theoretical_yield - actual_yield
        efficiency_factor = actual_yield / theoretical_yield
        
        return {
            'percent_yield': percent_yield,
            'yield_loss': yield_loss,
            'efficiency_factor': efficiency_factor,
            'theoretical_yield': theoretical_yield,
            'actual_yield': actual_yield
        }
    
    def calculate_concentration_from_moles(self, moles: float, volume_liters: float) -> float:
        """
        Calculate concentration (molarity) from moles and volume.
        
        Args:
            moles (float): Number of moles
            volume_liters (float): Volume in liters
            
        Returns:
            float: Concentration in mol/L (molarity)
            
        Raises:
            ValueError: If invalid data provided
        """
        if moles < 0:
            raise ValueError("Moles cannot be negative")
        
        if volume_liters <= 0:
            raise ValueError("Volume must be positive")
        
        concentration = moles / volume_liters
        return concentration
    
    def calculate_moles_from_concentration(self, concentration: float, 
                                        volume_liters: float) -> float:
        """
        Calculate moles from concentration and volume.
        
        Args:
            concentration (float): Concentration in mol/L
            volume_liters (float): Volume in liters
            
        Returns:
            float: Number of moles
            
        Raises:
            ValueError: If invalid data provided
        """
        if concentration < 0:
            raise ValueError("Concentration cannot be negative")
        
        if volume_liters <= 0:
            raise ValueError("Volume must be positive")
        
        moles = concentration * volume_liters
        return moles
    
    def get_stoichiometric_analysis(self, reactants: Dict[str, float], 
                                  products: Dict[str, float]) -> Dict[str, any]:
        """
        Get a comprehensive stoichiometric analysis.
        
        Args:
            reactants (Dict[str, float]): Dictionary of reactants and their amounts
            products (Dict[str, float]): Dictionary of products and their amounts
            
        Returns:
            Dict[str, any]: Comprehensive stoichiometric analysis
        """
        try:
            # Find limiting reactant
            limiting_reactant = self.find_limiting_reactant(reactants)
            
            # Calculate excess reactants
            excess_reactants = self.calculate_excess_reactant(reactants, limiting_reactant)
            
            # Calculate total moles
            total_reactant_moles = sum(reactants.values())
            total_product_moles = sum(products.values()) if products else 0
            
            # Calculate theoretical vs actual yield if products provided
            theoretical_yield = None
            actual_yield = None
            percent_yield = None
            
            if products and len(products) == 1:
                product_formula = list(products.keys())[0]
                actual_yield = products[product_formula]
                
                # For simplicity, assume theoretical yield equals limiting reactant amount
                theoretical_yield = reactants[limiting_reactant]
                percent_yield = self.calculate_percent_yield(theoretical_yield, actual_yield)
            
            analysis = {
                'limiting_reactant': limiting_reactant,
                'excess_reactants': excess_reactants,
                'total_reactant_moles': total_reactant_moles,
                'total_product_moles': total_product_moles,
                'theoretical_yield': theoretical_yield,
                'actual_yield': actual_yield,
                'percent_yield': percent_yield,
                'is_valid': True,
                'error': None
            }
            
            return analysis
            
        except ValueError as e:
            return {
                'limiting_reactant': None,
                'excess_reactants': {},
                'total_reactant_moles': 0,
                'total_product_moles': 0,
                'theoretical_yield': None,
                'actual_yield': None,
                'percent_yield': None,
                'is_valid': False,
                'error': str(e)
            } 