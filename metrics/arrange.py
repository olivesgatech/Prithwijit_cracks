import pandas as pd

# Function to parse the input text file
def parse_txt_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        model, test_set, finetune = None, None, None
        total_file_pairs, max_avg_dice, avg_dice, avg_bcd, avg_hausdorff = None, None, None, None, None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Evaluation for"):
                model = line.split(" ")[2]
            elif line.startswith("test set:"):
                test_set = line.split(": ")[1]
            elif line.startswith("finetune:"):
                finetune = line.split(": ")[1]
            elif line.startswith("Total file pairs for threshold selection:"):
                total_file_pairs = int(line.split(": ")[1])
            elif line.startswith("Max Average Dice Score:"):
                max_avg_dice = float(line.split(": ")[1])
            elif line.startswith("Best Threshold:"):
                best_threshold = float(line.split(": ")[1])
            elif line.startswith("Average Dice Score:"):
                avg_dice = float(line.split(": ")[1])
            elif line.startswith("Average BCD_2D:"):
                avg_bcd = float(line.split(": ")[1])
            elif line.startswith("Average hausdorff_distance_2D:"):
                avg_hausdorff = float(line.split(": ")[1])
                
                data.append([model, test_set, finetune, best_threshold, total_file_pairs, max_avg_dice, avg_dice, avg_bcd, avg_hausdorff])
                total_file_pairs, max_avg_dice, avg_dice, avg_bcd, avg_hausdorff = None, None, None, None, None

    columns = ["Model", "Test Set", "Finetune", "Best Threshold", "Total File Pairs", "Max Avg Dice Score", "Avg Dice Score", "Avg BCD_2D", "Avg Hausdorff Distance 2D"]
    return pd.DataFrame(data, columns=columns)

# Load and parse the text file
txt_file = "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/output_cracks_testing_segformer.txt"  # Path to the uploaded file
df = parse_txt_to_dataframe(txt_file)

# Save the DataFrame to a CSV file
df.to_csv("/home/prithwijit/Cracks/Prithwijit_cracks/metrics/output_cracks_testing_segformer.csv", index=False)

print("CSV file 'evaluation_results.csv' saved successfully.")
