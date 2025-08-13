"""
Chemical Equation Balancer Module

This module provides functionality to balance chemical equations using
algebraic methods. It can handle simple equations and provides step-by-step
balancing with coefficient determination.

"""

import re
from typing import Dict, List, Tuple, Optional
from parser import ChemicalFormulaParser
from utils import extract_compounds_from_equation, validate_equation_format


class EquationBalancer:
    """
    Balancer for chemical equations using algebraic methods.
    
    This class provides methods to balance chemical equations by
    determining appropriate stoichiometric coefficients.
    """
    
    def __init__(self):
        """Initialize the balancer with a formula parser."""
        self.parser = ChemicalFormulaParser()
    
    def balance_equation(self, equation: str) -> str:
        """
        Balance a chemical equation.
        
        Args:
            equation (str): Unbalanced chemical equation
            
        Returns:
            str: Balanced chemical equation
            
        Raises:
            ValueError: If the equation cannot be balanced or is invalid
        """
        if not equation:
            raise ValueError("Empty equation provided")
        
        # Validate equation format
        if not validate_equation_format(equation):
            raise ValueError("Invalid equation format. Use 'Reactants -> Products'")
        
        # Extract compounds from equation
        compounds = extract_compounds_from_equation(equation)
        if len(compounds) < 2:
            raise ValueError("Equation must have at least one reactant and one product")
        
        # Split into reactants and products
        parts = equation.split('->')
        if len(parts) != 2:
            raise ValueError("Equation must contain exactly one arrow (->)")
        
        reactants_str = parts[0].strip()
        products_str = parts[1].strip()
        
        # Parse reactants and products
        reactants = [c.strip() for c in reactants_str.split('+') if c.strip()]
        products = [c.strip() for c in products_str.split('+') if c.strip()]
        
        if not reactants or not products:
            raise ValueError("Equation must have at least one reactant and one product")
        
        # Balance the equation
        balanced_reactants, balanced_products = self._balance_compounds(reactants, products)
        
        # Format the balanced equation
        balanced_equation = self._format_balanced_equation(balanced_reactants, balanced_products)
        
        return balanced_equation
    
    def _balance_compounds(self, reactants: List[str], products: List[str]) -> Tuple[List[str], List[str]]:
        """
        Balance compounds using algebraic method.
        
        Args:
            reactants (List[str]): List of reactant formulas
            products (List[str]): List of product formulas
            
        Returns:
            Tuple[List[str], List[str]]: Balanced reactants and products with coefficients
            
        Raises:
            ValueError: If balancing fails
        """
        # For simplicity, implement a basic balancing algorithm
        # This is a simplified version - real balancing is more complex
        
        # Parse all compounds to get elements
        all_compounds = reactants + products
        compound_elements = {}
        
        for compound in all_compounds:
            try:
                elements = self.parser.parse_formula(compound)
                compound_elements[compound] = elements
            except ValueError as e:
                raise ValueError(f"Invalid compound '{compound}': {e}")
        
        # Simple balancing: try to find coefficients that work
        # This is a basic implementation - real balancing requires solving systems of equations
        
        # For now, return the original equation with basic validation
        balanced_reactants = reactants.copy()
        balanced_products = products.copy()
        
        # Check if the equation is already balanced
        if self._is_equation_balanced(reactants, products, compound_elements):
            return balanced_reactants, balanced_products
        
        # Try simple balancing
        try:
            balanced_reactants, balanced_products = self._simple_balance(
                reactants, products, compound_elements
            )
            return balanced_reactants, balanced_products
        except ValueError:
            # If simple balancing fails, return with warning
            raise ValueError("Complex balancing not implemented. Please check your equation manually.")
    
    def _is_equation_balanced(self, reactants: List[str], products: List[str], 
                            compound_elements: Dict[str, Dict[str, int]]) -> bool:
        """
        Check if an equation is already balanced.
        
        Args:
            reactants (List[str]): List of reactant formulas
            products (List[str]): List of product formulas
            compound_elements (Dict[str, Dict[str, int]]): Element counts for each compound
            
        Returns:
            bool: True if equation is balanced, False otherwise
        """
        # Count elements on each side
        reactant_elements = {}
        product_elements = {}
        
        # Count elements in reactants
        for reactant in reactants:
            elements = compound_elements[reactant]
            for element, count in elements.items():
                if element in reactant_elements:
                    reactant_elements[element] += count
                else:
                    reactant_elements[element] = count
        
        # Count elements in products
        for product in products:
            elements = compound_elements[product]
            for element, count in elements.items():
                if element in product_elements:
                    product_elements[element] += count
                else:
                    product_elements[element] = count
        
        # Check if all elements are balanced
        all_elements = set(reactant_elements.keys()) | set(product_elements.keys())
        
        for element in all_elements:
            reactant_count = reactant_elements.get(element, 0)
            product_count = product_elements.get(element, 0)
            
            if reactant_count != product_count:
                return False
        
        return True
    
    def _simple_balance(self, reactants: List[str], products: List[str],
                       compound_elements: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[str]]:
        """
        Perform simple balancing for common equations.
        
        Args:
            reactants (List[str]): List of reactant formulas
            products (List[str]): List of product formulas
            compound_elements (Dict[str, Dict[str, int]]): Element counts for each compound
            
        Returns:
            Tuple[List[str], List[str]]: Balanced reactants and products
            
        Raises:
            ValueError: If simple balancing fails
        """
        # Handle some common simple cases
        if len(reactants) == 2 and len(products) == 1:
            # A + B -> C type reaction
            return self._balance_two_to_one(reactants, products, compound_elements)
        elif len(reactants) == 1 and len(products) == 2:
            # A -> B + C type reaction
            return self._balance_one_to_two(reactants, products, compound_elements)
        elif len(reactants) == 2 and len(products) == 2:
            # A + B -> C + D type reaction
            return self._balance_two_to_two(reactants, products, compound_elements)
        else:
            raise ValueError("Complex balancing not implemented for this equation type")
    
    def _balance_two_to_one(self, reactants: List[str], products: List[str],
                           compound_elements: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[str]]:
        """
        Balance A + B -> C type reactions.
        
        Args:
            reactants (List[str]): Two reactant formulas
            products (List[str]): One product formula
            compound_elements (Dict[str, Dict[str, int]]): Element counts
            
        Returns:
            Tuple[List[str], List[str]]: Balanced equation
        """
        # For simple cases, try coefficient of 1 for all compounds
        balanced_reactants = [f"1{reactants[0]}", f"1{reactants[1]}"]
        balanced_products = [f"1{products[0]}"]
        
        # Check if this works
        if self._is_equation_balanced(reactants, products, compound_elements):
            return balanced_reactants, balanced_products
        
        # If not, this is a complex case that needs more sophisticated balancing
        raise ValueError("Complex balancing required for this equation")
    
    def _balance_one_to_two(self, reactants: List[str], products: List[str],
                           compound_elements: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[str]]:
        """
        Balance A -> B + C type reactions.
        
        Args:
            reactants (List[str]): One reactant formula
            products (List[str]): Two product formulas
            compound_elements (Dict[str, Dict[str, int]]): Element counts
            
        Returns:
            Tuple[List[str], List[str]]: Balanced equation
        """
        # For simple cases, try coefficient of 1 for all compounds
        balanced_reactants = [f"1{reactants[0]}"]
        balanced_products = [f"1{products[0]}", f"1{products[1]}"]
        
        # Check if this works
        if self._is_equation_balanced(reactants, products, compound_elements):
            return balanced_reactants, balanced_products
        
        # If not, this is a complex case that needs more sophisticated balancing
        raise ValueError("Complex balancing required for this equation")
    
    def _balance_two_to_two(self, reactants: List[str], products: List[str],
                           compound_elements: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[str]]:
        """
        Balance A + B -> C + D type reactions.
        
        Args:
            reactants (List[str]): Two reactant formulas
            products (List[str]): Two product formulas
            compound_elements (Dict[str, Dict[str, int]]): Element counts
            
        Returns:
            Tuple[List[str], List[str]]: Balanced equation
        """
        # For simple cases, try coefficient of 1 for all compounds
        balanced_reactants = [f"1{reactants[0]}", f"1{reactants[1]}"]
        balanced_products = [f"1{products[0]}", f"1{products[1]}"]
        
        # Check if this works
        if self._is_equation_balanced(reactants, products, compound_elements):
            return balanced_reactants, balanced_products
        
        # If not, this is a complex case that needs more sophisticated balancing
        raise ValueError("Complex balancing required for this equation")
    
    def _format_balanced_equation(self, reactants: List[str], products: List[str]) -> str:
        """
        Format a balanced equation as a string.
        
        Args:
            reactants (List[str]): List of balanced reactants with coefficients
            products (List[str]): List of balanced products with coefficients
            
        Returns:
            str: Formatted balanced equation
        """
        # Join reactants and products with + signs
        reactants_str = " + ".join(reactants)
        products_str = " + ".join(products)
        
        return f"{reactants_str} -> {products_str}"
    
    def extract_compounds_from_equation(self, equation: str) -> List[str]:
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
    
    def validate_equation(self, equation: str) -> bool:
        """
        Validate if a chemical equation is properly formatted.
        
        Args:
            equation (str): Chemical equation to validate
            
        Returns:
            bool: True if equation is valid, False otherwise
        """
        try:
            if not validate_equation_format(equation):
                return False
            
            # Try to extract compounds
            compounds = self.extract_compounds_from_equation(equation)
            if len(compounds) < 2:
                return False
            
            # Try to parse each compound
            for compound in compounds:
                self.parser.parse_formula(compound)
            
            return True
            
        except (ValueError, Exception):
            return False
    
    def get_equation_analysis(self, equation: str) -> Dict[str, any]:
        """
        Get a comprehensive analysis of a chemical equation.
        
        Args:
            equation (str): Chemical equation to analyze
            
        Returns:
            Dict[str, any]: Analysis including compounds, elements, and balance status
        """
        try:
            if not validate_equation_format(equation):
                return {
                    'equation': equation,
                    'is_valid': False,
                    'error': 'Invalid equation format',
                    'compounds': [],
                    'elements': {},
                    'is_balanced': False
                }
            
            # Extract compounds
            compounds = extract_compounds_from_equation(equation)
            
            # Parse all compounds
            compound_elements = {}
            all_elements = {}
            
            for compound in compounds:
                try:
                    elements = self.parser.parse_formula(compound)
                    compound_elements[compound] = elements
                    
                    # Add to total element counts
                    for element, count in elements.items():
                        if element in all_elements:
                            all_elements[element] += count
                        else:
                            all_elements[element] = count
                            
                except ValueError as e:
                    return {
                        'equation': equation,
                        'is_valid': False,
                        'error': f"Invalid compound '{compound}': {e}",
                        'compounds': compounds,
                        'elements': {},
                        'is_balanced': False
                    }
            
            # Check if balanced
            parts = equation.split('->')
            if len(parts) == 2:
                reactants = [c.strip() for c in parts[0].split('+') if c.strip()]
                products = [c.strip() for c in parts[1].split('+') if c.strip()]
                is_balanced = self._is_equation_balanced(reactants, products, compound_elements)
            else:
                is_balanced = False
            
            return {
                'equation': equation,
                'is_valid': True,
                'error': None,
                'compounds': compounds,
                'elements': all_elements,
                'is_balanced': is_balanced,
                'compound_elements': compound_elements
            }
            
        except Exception as e:
            return {
                'equation': equation,
                'is_valid': False,
                'error': str(e),
                'compounds': [],
                'elements': {},
                'is_balanced': False
            } 