from comparison_analysis import ComparisonAnalyzer


def main():
    print("\n" + "="*80)
    print("RAIN REMOVAL MODEL COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = ComparisonAnalyzer(
        results_dir="output/results",
        output_dir="output/comparisons"
    )
    
    # Run complete analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
