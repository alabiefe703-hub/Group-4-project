#!/usr/bin/env python3
"""
Chemical Analysis CLI Tool - Main Entry Point

This module serves as the main entry point for the Chemical Analysis CLI Tool.
It provides a user-friendly interface for performing various chemical calculations
and analysis tasks including formula parsing, molecular weight calculations,
equation balancing, stoichiometry, and concentration conversions.


"""

import sys
import os
from typing import Optional

# Import our custom modules
from parser import ChemicalFormulaParser
from molecular_calculator import MolecularCalculator
from equation_balancer import EquationBalancer
from stoichiometry import StoichiometryCalculator
from concentration_converter import ConcentrationConverter
from dimensional_analysis import DimensionalAnalyzer
from report_generator import ReportGenerator
from utils import clear_screen, validate_file_path


class ChemistryCLI:
    """
    Main CLI class that orchestrates all chemical analysis operations.
    
    This class provides a clean interface for users to interact with various
    chemical analysis tools through a menu-driven system.
    """
    
    def __init__(self):
        """Initialize the CLI with all necessary components."""
        self.parser = ChemicalFormulaParser()
        self.molecular_calc = MolecularCalculator()
        self.equation_balancer = EquationBalancer()
        self.stoichiometry_calc = StoichiometryCalculator()
        self.concentration_converter = ConcentrationConverter()
        self.dimensional_analyzer = DimensionalAnalyzer()
        self.report_generator = ReportGenerator()
        
    def display_welcome(self):
        """Display the welcome message and main menu."""
        clear_screen()
        print("Welcome to the Chemical Analysis CLI Tool")
        print("=" * 50)
        print("A comprehensive toolkit for chemical calculations and analysis")
        print("=" * 50)
        print()
        
    def display_menu(self):
        """Display the main menu options."""
        print("Available Operations:")
        print("1. Parse chemical formula")
        print("2. Calculate molecular weight")
        print("3. Balance chemical equation")
        print("4. Perform stoichiometry calculations")
        print("5. Convert concentration units")
        print("6. Generate chemistry report")
        print("7. Exit")
        print()
        
    def get_user_choice(self) -> str:
        """Get user's menu choice with validation."""
        while True:
            try:
                choice = input("Enter your choice (1-7): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6', '7']:
                    return choice
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
            except KeyboardInterrupt:
                print("\n Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n Goodbye!")
                sys.exit(0)
    
    def parse_formula(self):
        """Handle chemical formula parsing."""
        print("\n Chemical Formula Parser")
        print("-" * 30)
        
        # Get input method
        print("Choose input method:")
        print("1. Enter formula directly")
        print("2. Load from file")
        
        method = input("Enter choice (1-2): ").strip()
        
        if method == "1":
            formula = input("Enter chemical formula (e.g., H2O, C6H12O6): ").strip()
            if formula:
                try:
                    result = self.parser.parse_formula(formula)
                    print(f"\n Parsed formula: {formula}")
                    print(f"Elements: {result}")
                except ValueError as e:
                    print(f" Error: {e}")
        elif method == "2":
            file_path = input("Enter file path: ").strip()
            if validate_file_path(file_path):
                try:
                    formulas = self.parser.parse_file(file_path)
                    print(f"\n Parsed {len(formulas)} formulas from file")
                    for formula, elements in formulas.items():
                        print(f"  {formula}: {elements}")
                except Exception as e:
                    print(f" Error reading file: {e}")
            else:
                print(" Invalid file path")
        else:
            print(" Invalid choice")
    
    def calculate_molecular_weight(self):
        """Handle molecular weight calculations."""
        print("\n Molecular Weight Calculator")
        print("-" * 35)
        
        formula = input("Enter chemical formula: ").strip()
        if formula:
            try:
                # Parse the formula first
                elements = self.parser.parse_formula(formula)
                
                # Calculate molecular weight
                weight = self.molecular_calc.calculate_molecular_weight(elements)
                empirical = self.molecular_calc.get_empirical_formula(elements)
                
                print(f"\n Results for {formula}:")
                print(f"Molecular Weight: {weight:.2f} g/mol")
                print(f"Empirical Formula: {empirical}")
                
            except ValueError as e:
                print(f" Error: {e}")
    
    def balance_equation(self):
        """Handle chemical equation balancing."""
        print("\n Chemical Equation Balancer")
        print("-" * 35)
        
        print("Enter the unbalanced equation:")
        print("Format: Reactants -> Products")
        print("Example: H2 + O2 -> H2O")
        
        equation = input("Equation: ").strip()
        if equation:
            try:
                balanced = self.equation_balancer.balance_equation(equation)
                print(f"\n Balanced equation:")
                print(f"  {balanced}")
            except ValueError as e:
                print(f" Error: {e}")
    
    def perform_stoichiometry(self):
        """Handle stoichiometry calculations."""
        print("\n Stoichiometry Calculator")
        print("-" * 30)
        
        print("Choose calculation type:")
        print("1. Limiting reactant")
        print("2. Theoretical yield")
        print("3. Percent yield")
        
        calc_type = input("Enter choice (1-3): ").strip()
        
        if calc_type == "1":
            self._calculate_limiting_reactant()
        elif calc_type == "2":
            self._calculate_theoretical_yield()
        elif calc_type == "3":
            self._calculate_percent_yield()
        else:
            print(" Invalid choice")
    
    def _calculate_limiting_reactant(self):
        """Calculate limiting reactant."""
        print("\nLimiting Reactant Calculator")
        print("Enter reactants and their amounts:")
        
        reactants = {}
        while True:
            reactant = input("Enter reactant formula (or 'done'): ").strip()
            if reactant.lower() == 'done':
                break
            try:
                amount = float(input(f"Enter amount of {reactant} (moles): "))
                reactants[reactant] = amount
            except ValueError:
                print(" Invalid amount")
        
        if reactants:
            try:
                limiting = self.stoichiometry_calc.find_limiting_reactant(reactants)
                print(f"\n Limiting reactant: {limiting}")
            except ValueError as e:
                print(f" Error: {e}")
    
    def _calculate_theoretical_yield(self):
        """Calculate theoretical yield."""
        print("\n Theoretical Yield Calculator")
        
        reactant = input("Enter limiting reactant formula: ").strip()
        reactant_amount = input("Enter reactant amount (moles): ").strip()
        product = input("Enter product formula: ").strip()
        
        try:
            reactant_moles = float(reactant_amount)
            yield_amount = self.stoichiometry_calc.calculate_theoretical_yield(
                reactant, reactant_moles, product
            )
            print(f"\n Theoretical yield: {yield_amount:.2f} moles of {product}")
        except ValueError as e:
            print(f" Error: {e}")
    
    def _calculate_percent_yield(self):
        """Calculate percent yield."""
        print("\nPercent Yield Calculator")
        
        try:
            theoretical = float(input("Enter theoretical yield (moles): "))
            actual = float(input("Enter actual yield (moles): "))
            
            percent = self.stoichiometry_calc.calculate_percent_yield(theoretical, actual)
            print(f"\n Percent yield: {percent:.2f}%")
        except ValueError:
            print(" Invalid input values")
    
    def convert_concentration(self):
        """Handle concentration unit conversions."""
        print("\n Concentration Unit Converter")
        print("-" * 35)
        
        print("Available conversions:")
        print("1. Molarity to Molality")
        print("2. Molality to Molarity")
        print("3. Molarity to Normality")
        print("4. Normality to Molarity")
        
        conv_type = input("Enter choice (1-4): ").strip()
        
        try:
            value = float(input("Enter concentration value: "))
            solute_mw = float(input("Enter solute molecular weight (g/mol): "))
            
            if conv_type == "1":
                result = self.concentration_converter.molarity_to_molality(value, solute_mw)
                print(f" Molality: {result:.4f} mol/kg")
            elif conv_type == "2":
                result = self.concentration_converter.molality_to_molarity(value, solute_mw)
                print(f" Molarity: {result:.4f} mol/L")
            elif conv_type == "3":
                result = self.concentration_converter.molarity_to_normality(value, solute_mw)
                print(f" Normality: {result:.4f} N")
            elif conv_type == "4":
                result = self.concentration_converter.normality_to_molarity(value, solute_mw)
                print(f" Molarity: {result:.4f} mol/L")
            else:
                print(" Invalid choice")
                
        except ValueError:
            print(" Invalid input values")
    
    def generate_report(self):
        """Generate a comprehensive chemistry report."""
        print("\n Chemistry Report Generator")
        print("-" * 35)
        
        report_name = input("Enter report name (without extension): ").strip()
        if report_name:
            try:
                # Collect data for report
                formulas = input("Enter formulas to include (comma-separated): ").strip()
                if formulas:
                    formula_list = [f.strip() for f in formulas.split(',')]
                    self.report_generator.generate_report(report_name, formula_list)
                    print(f" Report generated: {report_name}.txt")
                else:
                    print(" No formulas provided")
            except Exception as e:
                print(f" Error generating report: {e}")
    
    def run(self):
        """Main CLI loop."""
        while True:
            try:
                self.display_welcome()
                self.display_menu()
                choice = self.get_user_choice()
                
                if choice == "1":
                    self.parse_formula()
                elif choice == "2":
                    self.calculate_molecular_weight()
                elif choice == "3":
                    self.balance_equation()
                elif choice == "4":
                    self.perform_stoichiometry()
                elif choice == "5":
                    self.convert_concentration()
                elif choice == "6":
                    self.generate_report()
                elif choice == "7":
                    print(" Thank you for using the Chemical Analysis CLI Tool!")
                    break
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f" Unexpected error: {e}")
                input("Press Enter to continue...")


def main():
    """Main entry point for the CLI application."""
    try:
        cli = ChemistryCLI()
        cli.run()
    except Exception as e:
        print(f" Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 