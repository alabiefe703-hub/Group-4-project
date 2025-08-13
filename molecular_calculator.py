"""
Molecular Calculator Module

This module provides functionality to calculate molecular weights and
empirical formulas for chemical compounds. It uses standard atomic weights
and provides comprehensive molecular analysis.


"""

import math
from typing import Dict, List, Tuple, Optional
from utils import calculate_gcd, simplify_ratio


class MolecularCalculator:
    """
    Calculator for molecular weights and empirical formulas.
    
    This class provides methods to calculate molecular weights using
    standard atomic weights and determine empirical formulas from
    molecular formulas.
    """
    
    def __init__(self):
        """Initialize the calculator with standard atomic weights."""
        # Standard atomic weights (g/mol) - simplified set for common elements
        self.atomic_weights = {
            'H': 1.008,    # Hydrogen
            'He': 4.003,   # Helium
            'Li': 6.941,   # Lithium
            'Be': 9.012,   # Beryllium
            'B': 10.811,   # Boron
            'C': 12.011,   # Carbon
            'N': 14.007,   # Nitrogen
            'O': 15.999,   # Oxygen
            'F': 18.998,   # Fluorine
            'Ne': 20.180,  # Neon
            'Na': 22.990,  # Sodium
            'Mg': 24.305,  # Magnesium
            'Al': 26.982,  # Aluminum
            'Si': 28.086,  # Silicon
            'P': 30.974,   # Phosphorus
            'S': 32.065,   # Sulfur
            'Cl': 35.453,  # Chlorine
            'Ar': 39.948,  # Argon
            'K': 39.098,   # Potassium
            'Ca': 40.078,  # Calcium
            'Sc': 44.956,  # Scandium
            'Ti': 47.867,  # Titanium
            'V': 50.942,   # Vanadium
            'Cr': 51.996,  # Chromium
            'Mn': 54.938,  # Manganese
            'Fe': 55.845,  # Iron
            'Co': 58.933,  # Cobalt
            'Ni': 58.693,  # Nickel
            'Cu': 63.546,  # Copper
            'Zn': 65.38,   # Zinc
            'Ga': 69.723,  # Gallium
            'Ge': 72.64,   # Germanium
            'As': 74.922,  # Arsenic
            'Se': 78.96,   # Selenium
            'Br': 79.904,  # Bromine
            'Kr': 83.80,   # Krypton
            'Rb': 85.468,  # Rubidium
            'Sr': 87.62,   # Strontium
            'Y': 88.906,   # Yttrium
            'Zr': 91.224,  # Zirconium
            'Nb': 92.906,  # Niobium
            'Mo': 95.94,   # Molybdenum
            'Tc': 98.0,    # Technetium
            'Ru': 101.07,  # Ruthenium
            'Rh': 102.906, # Rhodium
            'Pd': 106.42,  # Palladium
            'Ag': 107.868, # Silver
            'Cd': 112.411, # Cadmium
            'In': 114.818, # Indium
            'Sn': 118.710, # Tin
            'Sb': 121.760, # Antimony
            'Te': 127.60,  # Tellurium
            'I': 126.904,  # Iodine
            'Xe': 131.293, # Xenon
            'Cs': 132.905, # Cesium
            'Ba': 137.327, # Barium
            'La': 138.905, # Lanthanum
            'Ce': 140.116, # Cerium
            'Pr': 140.908, # Praseodymium
            'Nd': 144.242, # Neodymium
            'Pm': 145.0,   # Promethium
            'Sm': 150.36,  # Samarium
            'Eu': 151.964, # Europium
            'Gd': 157.25,  # Gadolinium
            'Tb': 158.925, # Terbium
            'Dy': 162.500, # Dysprosium
            'Ho': 164.930, # Holmium
            'Er': 167.259, # Erbium
            'Tm': 168.934, # Thulium
            'Yb': 173.04,  # Ytterbium
            'Lu': 174.967, # Lutetium
            'Hf': 178.49,  # Hafnium
            'Ta': 180.948, # Tantalum
            'W': 183.84,   # Tungsten
            'Re': 186.207, # Rhenium
            'Os': 190.23,  # Osmium
            'Ir': 192.217, # Iridium
            'Pt': 195.078, # Platinum
            'Au': 196.967, # Gold
            'Hg': 200.59,  # Mercury
            'Tl': 204.383, # Thallium
            'Pb': 207.2,   # Lead
            'Bi': 208.980, # Bismuth
            'Po': 209.0,   # Polonium
            'At': 210.0,   # Astatine
            'Rn': 222.0,   # Radon
            'Fr': 223.0,   # Francium
            'Ra': 226.0,   # Radium
            'Ac': 227.0,   # Actinium
            'Th': 232.038, # Thorium
            'Pa': 231.036, # Protactinium
            'U': 238.029,  # Uranium
            'Np': 237.0,   # Neptunium
            'Pu': 244.0,   # Plutonium
            'Am': 243.0,   # Americium
            'Cm': 247.0,   # Curium
            'Bk': 247.0,   # Berkelium
            'Cf': 251.0,   # Californium
            'Es': 252.0,   # Einsteinium
            'Fm': 257.0    # Fermium
        }
    
    def calculate_molecular_weight(self, elements: Dict[str, int]) -> float:
        """
        Calculate the molecular weight of a compound.
        
        Args:
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            float: Molecular weight in g/mol
            
        Raises:
            ValueError: If any element is not found in the atomic weights table
        """
        if not elements:
            raise ValueError("No elements provided for molecular weight calculation")
        
        total_weight = 0.0
        
        for element, count in elements.items():
            if element not in self.atomic_weights:
                raise ValueError(f"Unknown element: {element}")
            
            atomic_weight = self.atomic_weights[element]
            total_weight += atomic_weight * count
        
        return total_weight
    
    def get_empirical_formula(self, elements: Dict[str, int]) -> str:
        """
        Calculate the empirical formula from element counts.
        
        Args:
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            str: Empirical formula as a string
            
        Raises:
            ValueError: If no elements are provided
        """
        if not elements:
            raise ValueError("No elements provided for empirical formula calculation")
        
        # Find the greatest common divisor of all counts
        counts = list(elements.values())
        if not counts:
            raise ValueError("No element counts provided")
        
        # Calculate GCD of all counts
        gcd = counts[0]
        for count in counts[1:]:
            gcd = calculate_gcd(gcd, count)
        
        # Divide all counts by the GCD to get empirical formula
        empirical_elements = {}
        for element, count in elements.items():
            empirical_count = count // gcd
            if empirical_count > 0:
                empirical_elements[element] = empirical_count
        
        # Format the empirical formula
        return self._format_formula(empirical_elements)
    
    def _format_formula(self, elements: Dict[str, int]) -> str:
        """
        Format a formula from element dictionary.
        
        Args:
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            str: Formatted chemical formula
        """
        if not elements:
            return ""
        
        # Sort elements by symbol for consistent output
        sorted_elements = sorted(elements.items())
        
        formula_parts = []
        for element, count in sorted_elements:
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        
        return "".join(formula_parts)
    
    def calculate_percent_composition(self, elements: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate the percent composition by mass of each element.
        
        Args:
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            Dict[str, float]: Dictionary mapping element symbols to their percent composition
            
        Raises:
            ValueError: If molecular weight calculation fails
        """
        if not elements:
            raise ValueError("No elements provided for percent composition calculation")
        
        molecular_weight = self.calculate_molecular_weight(elements)
        
        percent_composition = {}
        for element, count in elements.items():
            atomic_weight = self.atomic_weights[element]
            element_mass = atomic_weight * count
            percent = (element_mass / molecular_weight) * 100
            percent_composition[element] = percent
        
        return percent_composition
    
    def calculate_moles_from_mass(self, mass: float, elements: Dict[str, int]) -> float:
        """
        Calculate the number of moles from a given mass.
        
        Args:
            mass (float): Mass in grams
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            float: Number of moles
            
        Raises:
            ValueError: If mass is negative or molecular weight calculation fails
        """
        if mass < 0:
            raise ValueError("Mass cannot be negative")
        
        molecular_weight = self.calculate_molecular_weight(elements)
        
        if molecular_weight <= 0:
            raise ValueError("Invalid molecular weight")
        
        return mass / molecular_weight
    
    def calculate_mass_from_moles(self, moles: float, elements: Dict[str, int]) -> float:
        """
        Calculate the mass from a given number of moles.
        
        Args:
            moles (float): Number of moles
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            float: Mass in grams
            
        Raises:
            ValueError: If moles is negative or molecular weight calculation fails
        """
        if moles < 0:
            raise ValueError("Moles cannot be negative")
        
        molecular_weight = self.calculate_molecular_weight(elements)
        
        if molecular_weight <= 0:
            raise ValueError("Invalid molecular weight")
        
        return moles * molecular_weight
    
    def get_molecular_analysis(self, elements: Dict[str, int]) -> Dict[str, any]:
        """
        Get a comprehensive molecular analysis.
        
        Args:
            elements (Dict[str, int]): Dictionary mapping element symbols to their counts
            
        Returns:
            Dict[str, any]: Comprehensive analysis including molecular weight, 
                           empirical formula, and percent composition
        """
        try:
            molecular_weight = self.calculate_molecular_weight(elements)
            empirical_formula = self.get_empirical_formula(elements)
            percent_composition = self.calculate_percent_composition(elements)
            
            analysis = {
                'molecular_weight': molecular_weight,
                'empirical_formula': empirical_formula,
                'percent_composition': percent_composition,
                'total_atoms': sum(elements.values()),
                'unique_elements': len(elements),
                'is_valid': True,
                'error': None
            }
            
            return analysis
            
        except ValueError as e:
            return {
                'molecular_weight': 0.0,
                'empirical_formula': "",
                'percent_composition': {},
                'total_atoms': 0,
                'unique_elements': 0,
                'is_valid': False,
                'error': str(e)
            }
    
    def validate_atomic_weight(self, element: str) -> bool:
        """
        Check if an element has a known atomic weight.
        
        Args:
            element (str): Element symbol to check
            
        Returns:
            bool: True if element has known atomic weight, False otherwise
        """
        return element in self.atomic_weights
    
    def get_atomic_weight(self, element: str) -> float:
        """
        Get the atomic weight of an element.
        
        Args:
            element (str): Element symbol
            
        Returns:
            float: Atomic weight in g/mol
            
        Raises:
            ValueError: If element is not found
        """
        if element not in self.atomic_weights:
            raise ValueError(f"Unknown element: {element}")
        
        return self.atomic_weights[element] 