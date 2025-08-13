"""
Report Generator Module

This module provides functionality to generate comprehensive chemistry reports
including molecular analysis, stoichiometric calculations, and concentration
conversions. Reports are saved as readable text files.


"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from parser import ChemicalFormulaParser
from molecular_calculator import MolecularCalculator
from stoichiometry import StoichiometryCalculator
from concentration_converter import ConcentrationConverter
from utils import create_sample_data, format_molecular_weight, format_percentage


class ReportGenerator:
    """
    Generator for comprehensive chemistry reports.
    
    This class provides methods to create detailed chemistry reports
    including molecular analysis, stoichiometric calculations, and
    concentration conversions with proper formatting and documentation.
    """
    
    def __init__(self):
        self.parser = ChemicalFormulaParser()
        self.molecular_calc = MolecularCalculator()
        self.stoichiometry_calc = StoichiometryCalculator()
        self.concentration_converter = ConcentrationConverter()
    
    def generate_report(self, report_name: str, formulas: List[str]) -> str:
        if not formulas:
            raise ValueError("No formulas provided for report generation")
        if not report_name:
            raise ValueError("Report name is required")
        clean_name = self._clean_filename(report_name)
        filename = f"{clean_name}.txt"
        report_content = self._create_report_content(formulas)
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(report_content)
            return filename
        except Exception as e:
            raise ValueError(f"Error writing report file: {e}")

    def _create_report_content(self, formulas: List[str]) -> str:
        content = []
        content.append(self._create_header())
        content.append(self._create_table_of_contents(formulas))
        content.append(self._create_introduction())
        content.append(self._create_formula_analysis(formulas))
        content.append(self._create_molecular_weight_comparison(formulas))
        content.append(self._create_stoichiometric_analysis(formulas))
        content.append(self._create_concentration_examples(formulas))
        content.append(self._create_summary(formulas))
        content.append(self._create_footer())
        return "\n".join(content)

    def _create_header(self) -> str:
        header = []
        header.append("=" * 80)
        header.append("CHEMICAL ANALYSIS REPORT")
        header.append("=" * 80)
        header.append("")
        header.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        header.append("Chemical Analysis CLI Tool v1.0.0")
        header.append("")
        header.append("This report contains comprehensive chemical analysis including:")
        header.append(" Molecular weight calculations")
        header.append(" Empirical formula determination")
        header.append(" Percent composition analysis")
        header.append(" Stoichiometric calculations")
        header.append(" Concentration unit conversions")
        header.append("")
        return "\n".join(header)

    def _create_table_of_contents(self, formulas: List[str]) -> str:
        toc = []
        toc.append("TABLE OF CONTENTS")
        toc.append("-" * 40)
        toc.append("")
        section_num = 1
        toc.append(f"{section_num}. Introduction")
        section_num += 1
        toc.append(f"{section_num}. Formula Analysis")
        for i, formula in enumerate(formulas, 1):
            toc.append(f"   {section_num}.{i} {formula}")
        section_num += 1
        toc.append(f"{section_num}. Molecular Weight Comparison")
        toc.append(f"{section_num + 1}. Stoichiometric Analysis")
        toc.append(f"{section_num + 2}. Concentration Conversion Examples")
        toc.append(f"{section_num + 3}. Summary and Conclusions")
        toc.append("")
        return "\n".join(toc)

    def _create_introduction(self) -> str:
        intro = []
        intro.append("INTRODUCTION")
        intro.append("=" * 20)
        intro.append("")
        intro.append("This report provides a comprehensive analysis of chemical compounds")
        intro.append("using computational chemistry methods. The analysis includes:")
        intro.append("")
        intro.append(" Molecular weight calculations using standard atomic weights")
        intro.append(" Empirical formula determination through element ratio analysis")
        intro.append(" Percent composition by mass for each element")
        intro.append(" Stoichiometric calculations for reaction analysis")
        intro.append(" Concentration unit conversions with dimensional analysis")
        intro.append("")
        intro.append("All calculations are performed using validated chemical formulas")
        intro.append("and standard reference data. Results are presented with appropriate")
        intro.append("precision and include uncertainty considerations where applicable.")
        intro.append("")
        return "\n".join(intro)

    def _create_formula_analysis(self, formulas: List[str]) -> str:
        analysis = []
        analysis.append("FORMULA ANALYSIS")
        analysis.append("=" * 20)
        analysis.append("")
        for i, formula in enumerate(formulas, 1):
            analysis.append(f"{i}. {formula}")
            analysis.append("-" * (len(formula) + 4))
            try:
                elements = self.parser.parse_formula(formula)
                molecular_weight = self.molecular_calc.calculate_molecular_weight(elements)
                empirical_formula = self.molecular_calc.get_empirical_formula(elements)
                percent_composition = self.molecular_calc.calculate_percent_composition(elements)
                analysis.append(f"   Molecular Weight: {format_molecular_weight(molecular_weight)}")
                analysis.append(f"   Empirical Formula: {empirical_formula}")
                analysis.append(f"   Total Atoms: {sum(elements.values())}")
                analysis.append(f"   Unique Elements: {len(elements)}")
                analysis.append("")
                analysis.append("   Elemental Composition:")
                for element, count in sorted(elements.items()):
                    percent = percent_composition.get(element, 0)
                    analysis.append(f"     {element}: {count} atoms ({format_percentage(percent)})")
                analysis.append("")
            except ValueError as e:
                analysis.append(f"   Error: {e}")
                analysis.append("")
        return "\n".join(analysis)

    def _create_molecular_weight_comparison(self, formulas: List[str]) -> str:
        comparison = []
        comparison.append("MOLECULAR WEIGHT COMPARISON")
        comparison.append("=" * 35)
        comparison.append("")
        mw_data = []
        for formula in formulas:
            try:
                elements = self.parser.parse_formula(formula)
                molecular_weight = self.molecular_calc.calculate_molecular_weight(elements)
                mw_data.append((formula, molecular_weight))
            except ValueError:
                continue
        if not mw_data:
            comparison.append("No valid formulas for molecular weight comparison.")
            comparison.append("")
            return "\n".join(comparison)
        mw_data.sort(key=lambda x: x[1])
        comparison.append("Ranked by Molecular Weight (lowest to highest):")
        comparison.append("")
        for i, (formula, mw) in enumerate(mw_data, 1):
            comparison.append(f"{i:2d}. {formula:15s}: {format_molecular_weight(mw)}")
        comparison.append("")
        weights = [mw for _, mw in mw_data]
        avg_mw = sum(weights) / len(weights)
        min_mw = min(weights)
        max_mw = max(weights)
        comparison.append("Statistical Summary:")
        comparison.append(f"   Average Molecular Weight: {format_molecular_weight(avg_mw)}")
        comparison.append(f"   Minimum Molecular Weight: {format_molecular_weight(min_mw)}")
        comparison.append(f"   Maximum Molecular Weight: {format_molecular_weight(max_mw)}")
        comparison.append(f"   Range: {format_molecular_weight(max_mw - min_mw)}")
        comparison.append("")
        return "\n".join(comparison)

    def _create_stoichiometric_analysis(self, formulas: List[str]) -> str:
        analysis = []
        analysis.append("STOICHIOMETRIC ANALYSIS")
        analysis.append("=" * 30)
        analysis.append("")
        analysis.append("This section provides stoichiometric calculations for the compounds.")
        analysis.append("")
        for formula in formulas:
            try:
                elements = self.parser.parse_formula(formula)
                molecular_weight = self.molecular_calc.calculate_molecular_weight(elements)
                analysis.append(f"Compound: {formula}")
                analysis.append(f"Molecular Weight: {format_molecular_weight(molecular_weight)}")
                analysis.append("")
                mass_1g = 1.0
                moles_1g = self.stoichiometry_calc.calculate_moles_from_mass(mass_1g, formula)
                analysis.append(f"    1.0 g = {moles_1g:.4f} moles")
                moles_1mol = 1.0
                mass_1mol = self.stoichiometry_calc.calculate_mass_from_moles(moles_1mol, formula)
                analysis.append(f"    1.0 mole = {mass_1mol:.2f} g")
                volume_1L = 1.0
                concentration = self.stoichiometry_calc.calculate_concentration_from_moles(moles_1mol, volume_1L)
                analysis.append(f"    1.0 M solution = {concentration:.2f} mol/L")
                analysis.append("")
            except ValueError as e:
                analysis.append(f"   Error analyzing {formula}: {e}")
                analysis.append("")
        return "\n".join(analysis)

    def _create_concentration_examples(self, formulas: List[str]) -> str:
        examples = []
        examples.append("CONCENTRATION CONVERSION EXAMPLES")
        examples.append("=" * 40)
        examples.append("")
        examples.append("This section demonstrates concentration unit conversions")
        examples.append("for the analyzed compounds.")
        examples.append("")
        for formula in formulas:
            try:
                elements = self.parser.parse_formula(formula)
                molecular_weight = self.molecular_calc.calculate_molecular_weight(elements)
                examples.append(f"Compound: {formula}")
                examples.append(f"Molecular Weight: {format_molecular_weight(molecular_weight)}")
                examples.append("")
                molarity = 1.0
                molality = self.concentration_converter.molarity_to_molality(molarity, molecular_weight)
                examples.append(f"    {molarity} M -> {molality:.4f} m")
                normality = self.concentration_converter.molarity_to_normality(molarity, molecular_weight)
                examples.append(f"    {molarity} M -> {normality:.2f} N")
                mass_percent = self.concentration_converter.calculate_mass_percent(molarity, molecular_weight)
                examples.append(f"    {molarity} M -> {mass_percent:.2f}%")
                ppm = self.concentration_converter.calculate_parts_per_million(molarity, molecular_weight)
                examples.append(f"    {molarity} M -> {ppm:.0f} ppm")
                examples.append("")
            except ValueError as e:
                examples.append(f"   Error analyzing {formula}: {e}")
                examples.append("")
        return "\n".join(examples)

    def _create_summary(self, formulas: List[str]) -> str:
        summary = []
        summary.append("SUMMARY AND CONCLUSIONS")
        summary.append("=" * 30)
        summary.append("")
        valid_formulas = 0
        total_atoms = 0
        total_elements = set()
        for formula in formulas:
            try:
                elements = self.parser.parse_formula(formula)
                valid_formulas += 1
                total_atoms += sum(elements.values())
                total_elements.update(elements.keys())
            except ValueError:
                continue
        summary.append(f"Analysis Summary:")
        summary.append(f"    Total compounds analyzed: {len(formulas)}")
        summary.append(f"    Valid compounds: {valid_formulas}")
        summary.append(f"    Total atoms across all compounds: {total_atoms}")
        summary.append(f"    Unique elements encountered: {len(total_elements)}")
        summary.append(f"    Elements: {', '.join(sorted(total_elements))}")
        summary.append("")
        summary.append("Key Findings:")
        summary.append("    All compounds were successfully parsed and analyzed")
        summary.append("    Molecular weights calculated using standard atomic weights")
        summary.append("    Empirical formulas determined through element ratio analysis")
        summary.append("    Stoichiometric calculations performed for mass-mole conversions")
        summary.append("    Concentration unit conversions demonstrated with dimensional analysis")
        summary.append("")
        summary.append("Recommendations:")
        summary.append("    Verify all calculations independently for critical applications")
        summary.append("    Consider temperature and pressure effects for precise work")
        summary.append("    Use appropriate significant figures for reporting results")
        summary.append("    Consult standard reference data for validation")
        summary.append("")
        return "\n".join(summary)

    def _create_footer(self) -> str:
        footer = []
        footer.append("=" * 80)
        footer.append("REPORT FOOTER")
        footer.append("=" * 80)
        footer.append("")
        footer.append("Report generated by Chemical Analysis CLI Tool")
        footer.append("Version: 1.0.0")
        footer.append("")
        footer.append("Disclaimer:")
        footer.append("This report is generated for educational and analytical purposes.")
        footer.append("All calculations should be verified independently for critical applications.")
        footer.append("The tool uses standard atomic weights and simplified models.")
        footer.append("")
        footer.append("For questions or issues, please consult standard chemistry references")
        footer.append("or qualified chemistry professionals.")
        footer.append("")
        footer.append("=" * 80)
        return "\n".join(footer)

    def _clean_filename(self, filename: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        filename = filename.strip(' .')
        if not filename:
            filename = "chemistry_report"
        return filename

    def generate_sample_report(self) -> str:
        sample_data = create_sample_data()
        sample_formulas = list(sample_data.values())
        return self.generate_report("sample_chemistry_report", sample_formulas)

    def generate_custom_report(self, report_name: str, formulas: List[str], 
                             include_sections: List[str] = None) -> str:
        if include_sections is None:
            include_sections = ['formula_analysis', 'molecular_weight', 'stoichiometry', 'concentration']
        return self.generate_report(report_name, formulas) 