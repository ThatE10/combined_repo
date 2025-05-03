def analyze_correlation(syco_eval_list, comparison_type="paired_vs_individual"):
    """
    Analyzes how well different evaluation methods correlate.
    
    Args:
        syco_eval_list: List of dictionaries containing evaluation results
        comparison_type: String indicating which comparison to perform
                        - "paired_vs_individual": Compare paired and individual scores
                        - "paired_vs_steered": Compare paired and steered scores
        
    Returns:
        Dictionary with mean errors and agreement statistics
    """
    # Initialize tracking variables
    base_errors = []
    suggestive_errors = []
    base_agreements = 0
    suggestive_agreements = 0
    valid_items = 0
    
    # Initialize separate lists for scores
    base_scores_paired = []
    suggestive_scores_paired = []
    base_scores_comparison = []
    suggestive_scores_comparison = []

    # Set up keys to look for based on comparison type
    if comparison_type == "paired_vs_individual":
        comparison_method = "judge_individual"
        comparison_name = "individual"
        required_keys = ['judge_paired', 'judge_individual']
    elif comparison_type == "paired_vs_steered":
        comparison_name = "steered"
        required_keys = ['judge_paired', 'judge_base_steered', 'judge_suggestive_steered']
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}")

    # Process each evaluation item
    for item in syco_eval_list:
        # Skip if missing required evaluation types
        if not all(key in item for key in required_keys):
            continue
            
        valid_items += 1
        
        # Extract paired scores
        paired_base_score = item['judge_paired']['base_response_score']
        paired_suggestive_score = item['judge_paired']['suggestive_response_score']
        
        # Extract comparison scores based on comparison type
        if comparison_type == "paired_vs_individual":
            comparison_base_score = item['judge_individual']['base_response_score']
            comparison_suggestive_score = item['judge_individual']['suggestive_response_score']
        else:  # paired_vs_steered
            comparison_base_score = item['judge_base_steered']['base_response_score']
            comparison_suggestive_score = item['judge_suggestive_steered']['suggestive_response_score']
        
        # Store scores in the appropriate lists
        base_scores_paired.append(paired_base_score)
        suggestive_scores_paired.append(paired_suggestive_score)
        base_scores_comparison.append(comparison_base_score)
        suggestive_scores_comparison.append(comparison_suggestive_score)
        
        # Calculate absolute errors between paired and comparison scores
        base_error = abs(paired_base_score - comparison_base_score)
        suggestive_error = abs(paired_suggestive_score - comparison_suggestive_score)
        
        # Track errors
        base_errors.append(base_error)
        suggestive_errors.append(suggestive_error)
        
        # Count perfect agreements (error = 0)
        if base_error == 0:
            base_agreements += 1
        
        if suggestive_error == 0:
            suggestive_agreements += 1
    
    # Compute means for paired scores
    mean_base_score_paired = sum(base_scores_paired) / len(base_scores_paired) if base_scores_paired else 0
    mean_suggestive_score_paired = sum(suggestive_scores_paired) / len(suggestive_scores_paired) if suggestive_scores_paired else 0
    
    # Compute means for comparison scores
    mean_base_score_comparison = sum(base_scores_comparison) / len(base_scores_comparison) if base_scores_comparison else 0
    mean_suggestive_score_comparison = sum(suggestive_scores_comparison) / len(suggestive_scores_comparison) if suggestive_scores_comparison else 0

    # Calculate mean errors
    mean_base_error = sum(base_errors) / len(base_errors) if base_errors else 0
    mean_suggestive_error = sum(suggestive_errors) / len(suggestive_errors) if suggestive_errors else 0
    
    # Calculate agreement rates
    base_agreement_rate = base_agreements / valid_items if valid_items > 0 else 0
    suggestive_agreement_rate = suggestive_agreements / valid_items if valid_items > 0 else 0
    
    # Compile and return results
    results = {
        'base_mean_paired': mean_base_score_paired,
        'suggestive_mean_paired': mean_suggestive_score_paired,
        f'base_mean_{comparison_name}': mean_base_score_comparison,
        f'suggestive_mean_{comparison_name}': mean_suggestive_score_comparison,
        'mean_base_error': mean_base_error,
        'mean_suggestive_error': mean_suggestive_error,
        'base_agreement_rate': base_agreement_rate,
        'suggestive_agreement_rate': suggestive_agreement_rate,
        'total_items_analyzed': valid_items,
        'comparison_type': comparison_type
    }
    
    return results

def display_correlation_results(results):
    """
    Displays the correlation analysis results in a readable format.
    """
    comparison_type = results['comparison_type']
    comparison_name = "individual" if comparison_type == "paired_vs_individual" else "steered"
    
    print(f"Sycophancy Evaluation: Paired vs {comparison_name.title()} Analysis")
    print("=" * 60)
    print(f"Total items analyzed: {results['total_items_analyzed']}")
    
    print("\nMean Scores:")
    print("  Paired Evaluation:")
    print(f"    Base Response: {results['base_mean_paired']:.2f}")
    print(f"    Suggestive Response: {results['suggestive_mean_paired']:.2f}")
    
    print(f"  {comparison_name.title()} Evaluation:")
    print(f"    Base Response: {results[f'base_mean_{comparison_name}']:.2f}")
    print(f"    Suggestive Response: {results[f'suggestive_mean_{comparison_name}']:.2f}")
    
    print(f"\nMean Errors (Difference between Paired and {comparison_name.title()}):")
    print(f"  Base Response: {results['mean_base_error']:.2f}")
    print(f"  Suggestive Response: {results['mean_suggestive_error']:.2f}")
    
    print("\nAgreement Rates (Percentage of exact matches):")
    print(f"  Base Response: {results['base_agreement_rate']*100:.1f}%")
    print(f"  Suggestive Response: {results['suggestive_agreement_rate']*100:.1f}%")

def evaluate_correlation(syco_eval_list, comparison_type="paired_vs_individual"):
    """
    Main function to run the correlation analysis and display results.
    
    Args:
        syco_eval_list: List of dictionaries containing evaluation results
        comparison_type: String indicating which comparison to perform
                        - "paired_vs_individual": Compare paired and individual scores
                        - "paired_vs_steered": Compare paired and steered scores
    
    Returns:
        Dictionary with correlation analysis results
    """
    results = analyze_correlation(syco_eval_list, comparison_type)
    display_correlation_results(results)
    return results

