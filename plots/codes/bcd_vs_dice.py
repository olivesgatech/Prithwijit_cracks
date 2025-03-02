import pandas as pd
import matplotlib.pyplot as plt

# File paths for each model
file_paths = {
    "deeplab": "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/result_csv/output_cracks_testing_deeplab.csv",
    "segformer": "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/result_csv/output_cracks_testing_segformer.csv",
    "unet": "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/result_csv/output_cracks_testing_unet.csv",
    "unetpp": "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/result_csv/output_cracks_testing_unetpp.csv"
}


# Define markers for different models
markers = {"deeplab": "o", "segformer": "s", "unet": "^", "unetpp": "d"}

# Define threshold type to filter
threshold_type = "BCD"

# Initialize plot
plt.figure(figsize=(8, 6))

# Collect all finetune strategies to assign colors
all_finetune_strategies = set()
for path in file_paths.values():
    df = pd.read_csv(path)
    all_finetune_strategies.update(df["Finetune"].unique())

# Assign colors to finetune strategies
colors = {strategy: color for strategy, color in zip(all_finetune_strategies, plt.cm.tab10.colors)}

for model, path in file_paths.items():
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Filter for the chosen threshold type
    df_filtered = df[df["Threshold Type"] == threshold_type].copy()
    
    # Normalize the BCD_2D and DICE values
    df_filtered.loc[:, "Normalized BCD_2D"] = (
        df_filtered["Average BCD_2D"] - df_filtered["Average BCD_2D"].min()
    ) / (df_filtered["Average BCD_2D"].max() - df_filtered["Average BCD_2D"].min())
    
    df_filtered.loc[:, "Normalized Hausdorff"] = (
        df_filtered["Average Hausdorff"] - df_filtered["Average Hausdorff"].min()
    ) / (df_filtered["Average Hausdorff"].max() - df_filtered["Average Hausdorff"].min())
    
    # Plot for each finetune strategy
    for finetune_strategy in df_filtered["Finetune"].unique():
        subset = df_filtered[df_filtered["Finetune"] == finetune_strategy]
        plt.scatter(
            subset["Normalized BCD_2D"], subset["Normalized Hausdorff"], 
            marker=markers[model], color=colors[finetune_strategy], label=f"{model} - {finetune_strategy}")

# Labels and title
plt.xlabel("Normalized Average BCD_2D")
plt.ylabel("Normalized Average Hausdorff")
plt.title(f"Normalized Finetune Strategies for Threshold Type: {threshold_type}")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

# Save the plot
plot_path = "/home/prithwijit/Cracks/Prithwijit_cracks/plots/cracks_2d_plot.png"
plt.savefig(plot_path, bbox_inches='tight')
# plt.show()

print(f"Plot saved at: {plot_path}")