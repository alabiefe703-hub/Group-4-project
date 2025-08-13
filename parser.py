"""
Chemical Formula Parser Module

This module provides functionality to parse chemical formulas and extract
element symbols and their counts. It handles various formula formats and
provides validation for chemical formulas.

"""

import re
import csv
from typing import Dict, List, Tuple, Optional
from utils import validate_chemical_symbol, format_chemical_formula


class ChemicalFormulaParser:
    """
    Parser for chemical formulas that extracts element symbols and counts.
    
    This class provides methods to parse chemical formulas from various
    input sources and extract the constituent elements and their quantities.
    """
    
    def __init__(self):
        """Initialize the parser with element validation patterns."""
        # Common element symbols (first letter uppercase, second lowercase if present)
        self.element_symbols = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
        }
        
        # Pattern to match element symbols and their counts
        # This pattern will match two-letter elements first, then single-letter elements
        self.element_pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
        
        # Create a more sophisticated parsing approach for two-letter elements
        # Use lookahead to ensure we don't match partial elements
        self.two_letter_pattern = re.compile(r'([A-Z][a-z])(\d*)')
        self.single_letter_pattern = re.compile(r'([A-Z])(\d*)')
        
        # Add common two-letter elements that might be missed
        self.two_letter_elements = {'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'}
    
    def parse_formula(self, formula: str) -> Dict[str, int]:
        """
        Parse a chemical formula and extract element counts.
        
        Args:
            formula (str): Chemical formula to parse (e.g., "H2O", "C6H12O6")
            
        Returns:
            Dict[str, int]: Dictionary mapping element symbols to their counts
            
        Raises:
            ValueError: If the formula is invalid or contains unrecognized elements
        """
        if not formula:
            raise ValueError("Empty formula provided")
        
        # Clean and format the formula
        formula = format_chemical_formula(formula)
        
        # Remove any spaces and parentheses for now (simplified parsing)
        formula = re.sub(r'\s+', '', formula)
        
        # Find all element symbols and their counts
        elements = {}
        position = 0
        
        while position < len(formula):
            # Try to match two-letter elements first
            if position + 1 < len(formula):
                potential_two_letter = formula[position:position + 2]
                if (formula[position].isupper() and 
                    formula[position + 1].islower() and 
                    self._is_valid_element(potential_two_letter)):
                    
                    element_symbol = potential_two_letter
                    position += 2
                    
                    # Look for count
                    count_str = ""
                    while position < len(formula) and formula[position].isdigit():
                        count_str += formula[position]
                        position += 1
                    
                    count = 1 if not count_str else int(count_str)
                    
                    # Add to elements dictionary
                    if element_symbol in elements:
                        elements[element_symbol] += count
                    else:
                        elements[element_symbol] = count
                    continue
            
            # Single-letter element
            if formula[position].isupper():
                element_symbol = formula[position]
                position += 1
                
                # Look for count
                count_str = ""
                while position < len(formula) and formula[position].isdigit():
                    count_str += formula[position]
                    position += 1
                
                # Validate the element symbol
                if not self._is_valid_element(element_symbol):
                    raise ValueError(f"Invalid element symbol: {element_symbol}")
                
                count = 1 if not count_str else int(count_str)
                
                # Add to elements dictionary
                if element_symbol in elements:
                    elements[element_symbol] += count
                else:
                    elements[element_symbol] = count
                continue
            
            # If no match found, check for invalid characters
            if formula[position].isalpha():
                raise ValueError(f"Invalid element symbol at position {position}: {formula[position]}")
            else:
                raise ValueError(f"Unexpected character at position {position}: {formula[position]}")
        
        if not elements:
            raise ValueError("No valid elements found in formula")
        
        return elements
    
    def _is_valid_element(self, symbol: str) -> bool:
        """
        Check if a symbol represents a valid chemical element.
        
        Args:
            symbol (str): Element symbol to validate
            
        Returns:
            bool: True if valid element, False otherwise
        """
        return symbol in self.element_symbols or symbol in self.two_letter_elements
    
    def parse_file(self, file_path: str) -> Dict[str, Dict[str, int]]:
        """
        Parse chemical formulas from a file.
        
        Args:
            file_path (str): Path to the file containing formulas
            
        Returns:
            Dict[str, Dict[str, int]]: Dictionary mapping formula strings to their parsed elements
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        formulas = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Try to detect if it's a CSV file
                if file_path.lower().endswith('.csv'):
                    reader = csv.reader(file)
                    for row in reader:
                        if len(row) >= 1:
                            formula = row[0].strip()
                            if formula:
                                try:
                                    elements = self.parse_formula(formula)
                                    formulas[formula] = elements
                                except ValueError as e:
                                    print(f"Warning: Skipping invalid formula '{formula}': {e}")
                else:
                    # Treat as plain text file
                    for line_num, line in enumerate(file, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            try:
                                elements = self.parse_formula(line)
                                formulas[line] = elements
                            except ValueError as e:
                                print(f"Warning: Skipping invalid formula on line {line_num}: {e}")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
        
        return formulas
    
    def validate_formula(self, formula: str) -> bool:
        """
        Validate if a chemical formula is properly formatted.
        
        Args:
            formula (str): Chemical formula to validate
            
        Returns:
            bool: True if formula is valid, False otherwise
        """
        try:
            self.parse_formula(formula)
            return True
        except ValueError:
            return False
    
    def get_formula_summary(self, formula: str) -> Dict[str, any]:
        """
        Get a comprehensive summary of a chemical formula.
        
        Args:
            formula (str): Chemical formula to analyze
            
        Returns:
            Dict[str, any]: Summary containing elements, counts, and validation info
        """
        try:
            elements = self.parse_formula(formula)
            
            summary = {
                'formula': formula,
                'elements': elements,
                'total_atoms': sum(elements.values()),
                'unique_elements': len(elements),
                'is_valid': True,
                'error': None
            }
            
            return summary
            
        except ValueError as e:
            return {
                'formula': formula,
                'elements': {},
                'total_atoms': 0,
                'unique_elements': 0,
                'is_valid': False,
                'error': str(e)
            }
    
    def extract_compounds_from_text(self, text: str) -> List[str]:
        """
        Extract potential chemical formulas from text.
        
        Args:
            text (str): Text containing potential chemical formulas
            
        Returns:
            List[str]: List of potential chemical formulas found in text
        """
        if not text:
            return []
        
        # Pattern to match potential chemical formulas
        # Looks for patterns like: ElementSymbol + optional number
        formula_pattern = re.compile(r'\b[A-Z][a-z]?\d*\b')
        
        potential_formulas = formula_pattern.findall(text)
        
        # Filter to only valid formulas
        valid_formulas = []
        for formula in potential_formulas:
            if self.validate_formula(formula):
                valid_formulas.append(formula)
        
        return valid_formulas
    
    def parse_complex_formula(self, formula: str) -> Dict[str, int]:
        """
        Parse complex formulas with parentheses and coefficients.
        
        Args:
            formula (str): Complex chemical formula (e.g., "Ca(OH)2")
            
        Returns:
            Dict[str, int]: Dictionary mapping element symbols to their counts
            
        Raises:
            ValueError: If the formula is invalid
        """
        if not formula:
            raise ValueError("Empty formula provided")
        
        # For now, implement a simplified version
        # This could be extended to handle parentheses and complex structures
        
        # Remove coefficients at the beginning
        formula = re.sub(r'^\d+', '', formula)
        
        # For simplicity, just parse as a basic formula
        return self.parse_formula(formula)
    
    def format_elements_dict(self, elements: Dict[str, int]) -> str:
        """
        Format an elements dictionary as a readable string.
        
        Args:
            elements (Dict[str, int]): Dictionary of elements and counts
            
        Returns:
            str: Formatted string representation
        """
        if not elements:
            return "No elements"
        
        formatted_parts = []
        for element, count in sorted(elements.items()):
            if count == 1:
                formatted_parts.append(element)
            else:
                formatted_parts.append(f"{element}{count}")
        
        return " ".join(formatted_parts) 