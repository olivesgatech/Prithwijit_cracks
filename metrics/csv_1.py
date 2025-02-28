txt_file_path = "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/output_cracks_testing_deeplab.txt"
modified_txt_file_path = "/home/prithwijit/Cracks/Prithwijit_cracks/metrics/cracks/modified_output_cracks_testing_deeplab.txt"

# Read the original text file
with open(txt_file_path, "r") as file:
    lines = file.readlines()

threshold_count = 0  # Counter to track occurrences of "Best Threshold"

# Modify the lines
modified_lines = []
for line in lines:
    if "Best Threshold" in line:
        threshold_count += 1
        if threshold_count % 3 == 2:  # (n+1)th occurrence (odd)
            line = line.replace("Best Threshold", "Best Threshold for DICE")
        elif threshold_count % 3 == 0:  # (n+2)th occurrence (even)
            line = line.replace("Best Threshold", "Best Threshold for Hausdorff")
    modified_lines.append(line)

# Save the modified content to a new file
with open(modified_txt_file_path, "w") as file:
    file.writelines(modified_lines)

print(f"Modified text file saved as {modified_txt_file_path}")
