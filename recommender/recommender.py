import pandas as pd
import numpy as np

# ==========================================
# 1. Rank References Function
# ==========================================
voxel_diff_weight=0.0001
def rank_references(input_dict, references):
    ranked_refs = []
    
    for ref in references:
        # Create a copy so we don't modify the original references list
        current_ref = ref.copy()
        
        # 1.3 Calculate the absolute difference of "VoxelCount"
        voxeldiff = abs(current_ref["VoxelCount"] - input_dict["VoxelCount"])
        
        # 1.4 Mark "same_physician" as 1 or 0
        if current_ref["Physician"] == input_dict["Physician"]:
            same_physician = 1
        else:
            same_physician = 0
            
        # 1.5 Calculate score
        # Note: With a '+', larger voxel differences give higher scores. 
        # Consider using 'same_physician - weight * voxeldiff' if you want closer voxel counts to rank higher.
        score = same_physician - voxel_diff_weight * voxeldiff
        
        current_ref["same_physician"] = same_physician
        current_ref["voxeldiff"] = voxeldiff
        current_ref["score"] = score
        
        ranked_refs.append(current_ref)
        
    # 1.6 Return sorted references by score
    # Sorting descending (reverse=True) so that same_physician=1 comes first.
    ranked_refs = sorted(ranked_refs, key=lambda x: x["score"], reverse=True)
    
    return ranked_refs

# ==========================================
# 2. Test Recommender Function
# ==========================================
def test_recommender(test_table, reference_table, results_table, output_path):
    # 2.1 Load specific sheets from the Excel files
    test_df = pd.read_excel(test_table, sheet_name="test top1")
    ref_df = pd.read_excel(reference_table, sheet_name="10 cluster centroids")
    res_df = pd.read_excel(results_table, sheet_name="All predictions")
    
    # 2.2 Make a list of test dicts with MRN, Physician, VoxelCount
    test_dicts = test_df[["MRN", "Physician", "VoxelCount"]].to_dict('records')
    test_dicts = [dict for dict in test_dicts if dict['MRN'] is not np.nan]
    # 2.3 Make a list of reference dicts with MRN, Physician, VoxelCount
    ref_dicts = ref_df[["MRN", "Physician", "VoxelCount"]].to_dict('records')
    
    # Index the results dataframe by the 'Label' column for much faster lookups in step 2.5
    res_df.set_index("Label", inplace=True)
    
    # Dictionaries to store the metrics for calculating averages later
    top1_metrics = {"Dice": [], "HD95": [], "VOE": []}
    top5_metrics = {"Dice": [], "HD95": [], "VOE": []}
    all_metrics = {"Dice": [], "HD95": [], "VOE": []}
    
    # 2.4 Iterate through each test dictionary
    for t_dict in test_dicts:
        test_mrn = t_dict["MRN"]
        
        # Call the ranking function
        ranked_refs = rank_references(t_dict, ref_dicts)
        
        # Lists to hold the metrics for the current test MRN across all ranked references
        t_dice, t_hd95, t_voe = [], [], []
        
        # 2.5 Search for results based on the label format
        for ref in ranked_refs:
            ref_mrn = ref["MRN"]
            label = f"{test_mrn}_from_{ref_mrn}.npz"
            
            if label in res_df.index:
                row = res_df.loc[label]
                # If there are duplicate labels, just take the first one
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                    
                t_dice.append(row["Dice"])
                t_hd95.append(row["HD95"])
                t_voe.append(row["VOE"])
            else:
                # If no matching npz file is found, append NaN to avoid breaking the math
                t_dice.append(np.nan)
                t_hd95.append(np.nan)
                t_voe.append(np.nan)
                
        # 2.6 Aggregate the metrics
        # For "best", I am assuming Higher is better for Dice, and Lower is better for HD95/VOE.
        
        # Top 1 
        if len(t_dice) > 0:
            top1_metrics["Dice"].append(t_dice[0])
            top1_metrics["HD95"].append(t_hd95[0])
            top1_metrics["VOE"].append(t_voe[0])
            
        # Best in Top 5
        t5_dice, t5_hd95, t5_voe = t_dice[:5], t_hd95[:5], t_voe[:5]
        if not np.all(np.isnan(t5_dice)):  # Ensure there's at least one valid number
            top5_metrics["Dice"].append(np.nanmax(t5_dice))
            top5_metrics["HD95"].append(np.nanmin(t5_hd95))
            top5_metrics["VOE"].append(np.nanmin(t5_voe))
            
        # Best in All Rankings
        if not np.all(np.isnan(t_dice)):
            all_metrics["Dice"].append(np.nanmax(t_dice))
            all_metrics["HD95"].append(np.nanmin(t_hd95))
            all_metrics["VOE"].append(np.nanmin(t_voe))
            
    # Calculate the final averages across all test MRNs
    avg_results = {
        "Metric": ["Dice", "HD95", "VOE"],
        "Top-1 Average": [
            np.nanmean(top1_metrics["Dice"]),
            np.nanmean(top1_metrics["HD95"]),
            np.nanmean(top1_metrics["VOE"])
        ],
        "Top-5 Best Average": [
            np.nanmean(top5_metrics["Dice"]),
            np.nanmean(top5_metrics["HD95"]),
            np.nanmean(top5_metrics["VOE"])
        ],
        "All Best Average": [
            np.nanmean(all_metrics["Dice"]),
            np.nanmean(all_metrics["HD95"]),
            np.nanmean(all_metrics["VOE"])
        ]
    }
    
    # 2.7 Save results to another excel file
    output_df = pd.DataFrame(avg_results)
    output_df.to_excel(output_path, index=False)
    
    # Return the Dice metrics as requested
    top1_dice = avg_results["Top-1 Average"][0]
    top5_dice = avg_results["Top-5 Best Average"][0]
    top_all_dice = avg_results["All Best Average"][0]
    
    return top1_dice, top5_dice, top_all_dice

print(f'voxel_diff_weight is {voxel_diff_weight}')
top1_dice, top5_dice, top_all_dice=test_recommender(...)
print(f'top1_dice:{top1_dice},\n top5_dice:{top5_dice},\n top_all_dice:{top_all_dice}')

