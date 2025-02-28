import pandas as pd

# Define input and output file paths
txt_file_path = "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/output_cracks_testing_deeplab.txt"
csv_file_path = "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/output_cracks_testing_deeplab.csv"


# Initialize list to store data
data = []

# Read the text file and extract relevant information
with open(txt_file_path, "r") as file:
    lines = [line.strip() for line in file.readlines() if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("Evaluation for"):
            model = lines[i].split(" ")[2].strip()
            
            if i+1 < len(lines) and ": " in lines[i+1]:
                test_set = lines[i+1].split(": ")[1].strip()
            else:
                test_set = "Unknown"
            
            if i+2 < len(lines) and ": " in lines[i+2]:
                finetune = lines[i+2].split(": ")[1].strip()
            else:
                finetune = "Unknown"
            
            j = 3
            while i + j < len(lines) and "Threshold" in lines[i + j]:
                threshold_type = lines[i + j].split(" ")[-1].strip()
                best_threshold = float(lines[i + j].split(": ")[1].strip())
                avg_bcd_2d = float(lines[i + j + 1].split(": ")[1].strip())
                avg_dice = float(lines[i + j + 2].split(": ")[1].strip())
                avg_hausdorff = float(lines[i + j + 3].split(": ")[1].strip())
                
                data.append([model, test_set, finetune, threshold_type, best_threshold, avg_bcd_2d, avg_dice, avg_hausdorff])
                j += 4  # Move to the next threshold block
            
            i += j  # Move to next evaluation block
        else:
            i += 1

# Defining column names
columns = ["Model", "Test Set", "Finetune", "Threshold Type", "Best Threshold",
           "Average BCD_2D", "Average DICE", "Average Hausdorff"]

# Creating the dataframe
df = pd.DataFrame(data, columns=columns)

# Saving to CSV
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved as {csv_file_path}")
