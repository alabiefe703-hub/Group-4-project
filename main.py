
"""
Compact Electoral Analysis Tool
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
import argparse

class CompactElectoralAnalyzer:
    def __init__(self):
        self.data = None
        self.results = {}
    
    def load_data(self, filename):
        """Load data from CSV/Excel"""
        try:
            self.data = pd.read_csv(filename) if filename.endswith('.csv') else pd.read_excel(filename)
            print(f"Data loaded: {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    def analyze(self):
        """Run all analyses"""
        if self.data is None:
            print("No data loaded")
            return
        
        analyses = [
            self._turnout_analysis,
            self._voting_patterns,
            self._detect_anomalies,
            self._ballot_validation,
            self._boundary_analysis
        ]
        
        for analysis in analyses:
            try:
                analysis()
            except Exception as e:
                print(f"Analysis failed: {str(e)}")
        
        self._generate_report()
    
    def _turnout_analysis(self):
        """Calculate turnout statistics"""
        self.data['turnout'] = self.data['total_votes'] / self.data['eligible_voters'] * 100
        self.results['turnout'] = {
            'national': self.data['turnout'].mean(),
            'regional': self.data.groupby('region')['turnout'].agg(['mean', 'std']).to_dict(),
            'extremes': {
                'highest': self.data.nlargest(3, 'turnout')[['constituency_name', 'turnout']].values.tolist(),
                'lowest': self.data.nsmallest(3, 'turnout')[['constituency_name', 'turnout']].values.tolist()
            }
        }
    
    def _voting_patterns(self):
        """Analyze voting patterns"""
        candidates = [c for c in self.data.columns if c.startswith('candidate_')]
        for c in candidates:
            self.data[f'{c}_share'] = self.data[c] / self.data['total_votes'] * 100
        
        self.results['voting_patterns'] = {
            'candidates': {
                c: {
                    'total': int(self.data[c].sum()),
                    'share': float(self.data[f'{c}_share'].mean()),
                    'wins': int((self.data[c] == self.data[candidates].max(axis=1)).sum())
                } for c in candidates
            },
            'invalid_votes': {
                'mean': float((self.data['invalid_votes'] / self.data['total_votes'] * 100).mean()),
                'high': self.data.nlargest(3, 'invalid_votes')[['constituency_name', 'invalid_votes']].values.tolist()
            }
        }
    
    def _detect_anomalies(self):
        """Detect statistical anomalies"""
        # Benford's Law
        digits = self.data['total_votes'].astype(str).str[0].astype(int)
        observed = digits.value_counts().sort_index()
        expected = np.log10(1 + 1/np.arange(1, 10)) * len(self.data)
        _, p = stats.chisquare(observed, expected)
        
        # Turnout anomalies
        clf = IsolationForest(contamination=0.05)
        anomalies = clf.fit_predict(self.data[['turnout', 'population_density']].fillna(0))
        
        self.results['anomalies'] = {
            'benford': {'p_value': p, 'anomaly': p < 0.05},
            'turnout': self.data.iloc[np.where(anomalies == -1)[0]][['constituency_name', 'turnout']].values.tolist()
        }
    
    def _ballot_validation(self):
        """Validate ballot counts"""
        candidates = [c for c in self.data.columns if c.startswith('candidate_')]
        discrepancies = self.data[candidates].sum(axis=1) + self.data['invalid_votes'] - self.data['total_votes']
        self.results['validation'] = {
            'discrepancies': sum(discrepancies != 0),
            'details': self.data[discrepancies != 0][['constituency_name', 'total_votes']].values.tolist()
        }
    
    def _boundary_analysis(self):
        """Analyze constituency boundaries"""
        if 'population_density' in self.data.columns:
            self.data['voting_power'] = self.data['eligible_voters'] / self.data['population_density']
            self.results['boundaries'] = {
                'malapportioned': self.data[
                    (self.data['eligible_voters'] > self.data['eligible_voters'].mean() * 1.5) |
                    (self.data['eligible_voters'] < self.data['eligible_voters'].mean() * 0.5)
                ][['constituency_name', 'eligible_voters']].values.tolist(),
                'voting_power': self.data['voting_power'].describe().to_dict()
            }
    
    def _generate_report(self):
        """Generate analysis report"""
        print("\n=== ELECTION ANALYSIS REPORT ===")
        print(f"National Turnout: {self.results['turnout']['national']:.1f}%")
        
        print("\nTop Candidates:")
        for cand, stats in self.results['voting_patterns']['candidates'].items():
            print(f"- {cand}: {stats['total']:,} votes ({stats['share']:.1f}%)")
        
        print("\nAnomalies Detected:")
        print(f"- Benford's Law: {'Yes' if self.results['anomalies']['benford']['anomaly'] else 'No'}")
        if self.results['anomalies']['turnout']:
            print("- Unusual turnout in:", ', '.join([x[0] for x in self.results['anomalies']['turnout']]))
        
        if self.results['validation']['discrepancies']:
            print(f"\nWARNING: {self.results['validation']['discrepancies']} ballot count discrepancies found")

def main():
    analyzer = CompactElectoralAnalyzer()
    if analyzer.load_data("election_data.csv"):  # Fixed filename
        analyzer.analyze()

if __name__ == "__main__":
    main()