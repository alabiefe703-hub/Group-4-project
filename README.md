# Chemical Analysis CLI Tool

A comprehensive, modular Python CLI application for performing chemical analysis tasks including formula parsing, molecular weight calculations, equation balancing, stoichiometry, and concentration conversions.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Documentation](#technical-documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Chemical Analysis CLI Tool is designed to help students, educators, and researchers perform various chemical calculations and analysis tasks. It provides a user-friendly command-line interface with comprehensive functionality for:

- Chemical Formula Parsing: Parse and validate chemical formulas
- Molecular Weight Calculations: Calculate molecular weights and empirical formulas
- Equation Balancing: Balance chemical equations using algebraic methods
- Stoichiometry: Perform limiting reactant, theoretical yield, and percent yield calculations
- Concentration Conversions: Convert between molarity, molality, normality, and other units
- Dimensional Analysis: Step-by-step unit analysis and validation
- Report Generation: Create comprehensive chemistry reports

## Features

### Chemical Formula Parser
- Parse chemical formulas from text or file input
- Extract element symbols and their counts
- Validate formula syntax and element symbols
- Support for common chemical elements

### Molecular Weight Calculator
- Calculate molecular weights using standard atomic weights
- Determine empirical formulas from molecular formulas
- Calculate percent composition by mass
- Convert between mass and moles

### Equation Balancer
- Balance chemical equations using algebraic methods
- Support for simple and complex reactions
- Validate equation format and compound formulas
- Provide step-by-step balancing analysis

### Stoichiometry Calculator
- Find limiting reactants in reactions
- Calculate theoretical and actual yields
- Determine percent yield and reaction efficiency
- Perform mass-mole conversions

### Concentration Converter
- Convert between molarity (M), molality (m), and normality (N)
- Calculate mass percent and parts per million
- Support for different solvents with known densities
- Include dimensional analysis validation

### Dimensional Analysis
- Step-by-step unit conversion analysis
- Validate calculations using dimensional analysis
- Provide educational explanations for conversions
- Support for common chemical units

### Report Generator
- Generate comprehensive chemistry reports
- Include molecular analysis, stoichiometry, and conversions
- Export results to readable text files
- Provide statistical summaries and comparisons

## Installation

### Prerequisites
- Python 3.8 or higher
- No external dependencies required (uses only standard library)

### Setup
1. Clone or download the project files
2. Navigate to the ChemistryTool directory
3. Run the application:

```bash
python main.py
```

## Usage

### Starting the Application
```bash
cd ChemistryTool
python main.py
```

### Main Menu Options

1. Parse chemical formula - Parse and analyze chemical formulas
2. Calculate molecular weight - Calculate molecular weights and empirical formulas
3. Balance chemical equation - Balance chemical equations
4. Perform stoichiometry - Calculate limiting reactants, yields, and efficiency
5. Convert concentration units - Convert between concentration units
6. Generate chemistry report - Create comprehensive reports
7. Exit - Exit the application

### Example Usage

#### Parsing Chemical Formulas
```
Enter chemical formula: H2O
Parsed formula: H2O
Elements: {'H': 2, 'O': 1}
```

#### Calculating Molecular Weight
```
Enter chemical formula: C6H12O6
Results for C6H12O6:
Molecular Weight: 180.16 g/mol
Empirical Formula: CH2O
```

#### Balancing Equations
```
Enter equation: H2 + O2 -> H2O
Balanced equation: 2H2 + O2 -> 2H2O
```

#### Concentration Conversions
```
Enter concentration value: 1.0
Enter solute molecular weight: 58.44
Molality: 1.002 mol/kg
```

## Technical Documentation

### Project Structure
```
ChemistryTool/
├── main.py                          # CLI entry point and command routing
├── parser.py                        # Chemical formula parser
├── molecular_calculator.py          # Empirical and molecular weight calculations
├── equation_balancer.py             # Equation balancing logic
├── stoichiometry.py                 # Yield, limiting reactant, and mole-mass conversions
├── concentration_converter.py       # Molarity, molality, normality conversions
├── dimensional_analysis.py          # Stepwise unit analysis and validation
├── report_generator.py              # Summary report output
├── utils.py                         # Reusable helper functions
├── data/                            # Sample data files
│   ├── sample_equations.txt         # Sample equations and formulas
│   └── atomic_weights.csv           # Element weights table
├── test/                            # Unit tests
│   ├── test_parser.py
│   ├── test_equation_balancer.py
│   └── test_concentration_converter.py
└── README.md                        # Documentation
```

### Module Descriptions

#### main.py
- Main CLI interface and command routing
- User input handling and menu system
- Integration of all modules

#### parser.py
- Chemical formula parsing and validation
- Element symbol recognition and counting
- File input processing (TXT, CSV)

#### molecular_calculator.py
- Molecular weight calculations using atomic weights
- Empirical formula determination
- Percent composition analysis
- Mass-mole conversions

#### equation_balancer.py
- Chemical equation balancing using algebraic methods
- Equation validation and compound extraction
- Simple balancing algorithms for common reaction types

#### stoichiometry.py
- Limiting reactant determination
- Theoretical and actual yield calculations
- Percent yield and reaction efficiency analysis
- Concentration calculations

#### concentration_converter.py
- Unit conversions between molarity, molality, normality
- Mass percent and parts per million calculations
- Solvent density considerations
- Dimensional analysis validation

#### dimensional_analysis.py
- Step-by-step unit conversion analysis
- Unit validation and compatibility checking
- Educational explanations for conversions
- Common unit conversion guides

#### report_generator.py
- Comprehensive chemistry report generation
- Multiple analysis sections and summaries
- Text file export with proper formatting
- Statistical analysis and comparisons

#### utils.py
- Helper functions for file validation
- Screen clearing and formatting utilities
- Number parsing and validation
- Chemical formula formatting

### Data Files

#### data/sample_equations.txt
- Sample chemical equations for testing
- Various reaction types (synthesis, decomposition, redox)
- Properly formatted equations for validation

#### data/atomic_weights.csv
- Comprehensive atomic weights table
- Standard atomic weights for all elements
- CSV format for easy parsing

### Error Handling
The application includes comprehensive error handling:
- Input validation for all user inputs
- Graceful handling of invalid formulas and equations
- Clear error messages with suggestions for correction
- Fallback mechanisms for complex calculations

### Educational Features
- Step-by-step explanations for all calculations
- Dimensional analysis for unit conversions
- Validation checks with educational feedback
- Comprehensive reporting with explanations

## Testing

### Running Tests
```bash
cd ChemistryTool
python -m unittest discover test/
```

### Test Coverage
- Parser Tests: Formula parsing, validation, and error handling
- Equation Balancer Tests: Equation validation and balancing
- Concentration Converter Tests: Unit conversions and validation
- Integration Tests: End-to-end functionality testing

### Test Files
- test/test_parser.py - Chemical formula parser tests
- test/test_equation_balancer.py - Equation balancing tests
- test/test_concentration_converter.py - Concentration conversion tests

## Contributing

### Development Guidelines
1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings and comments
3. Include unit tests for new functionality
4. Update documentation as needed
5. Use only standard library modules

### Code Style
- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Include type hints where appropriate
- Add inline comments for complex calculations

### Testing Guidelines
- Write unit tests for all new functions
- Include edge cases and error conditions
- Test both valid and invalid inputs
- Ensure good test coverage

## License

This project is designed for educational purposes. All calculations should be verified independently for critical applications.

## Disclaimer

This tool is designed for educational and analytical purposes. All calculations should be verified independently for critical applications. The tool uses standard atomic weights and simplified models. For questions or issues, please consult standard chemistry references or qualified chemistry professionals.

## Support

For questions, issues, or suggestions:
1. Check the documentation and examples
2. Review the error messages for guidance
3. Verify input formats and units
4. Consult standard chemistry references

---

Chemical Analysis CLI Tool v1.0.0
Empowering chemical education through computational analysis 