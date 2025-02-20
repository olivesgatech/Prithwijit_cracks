# import os



# import numpy as np

# from skimage.transform import resize




# thebe_path = '/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_expert/logits/'

# out_path =  '/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_expert/logits_stitched/'


# if not os.path.exists(out_path):
#     os.makedirs(out_path)

# files = os.listdir(thebe_path)

# for i in range(int(len(files)/2)):



#     top = np.load(thebe_path + 'array_' +  str(2*i) + '.npy').T

#     bottom = np.load(thebe_path + 'array_' + str(2*i+1) + '.npy').T



#     top = resize(top, (1588,1588), order=0, mode='edge', anti_aliasing=False)

#     bottom = resize(bottom, (1588,1588), order=0, mode='edge', anti_aliasing=False)



#     # Stack images vertically

#     stitched_img = np.vstack((bottom, top))



#     final_img = stitched_img[:3174, :1537]



#     np.save(out_path + 'array_' + str(i) + '.npy', final_img)


import os
import numpy as np
from skimage.transform import resize
import argparse

def process_and_stitch(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    files = sorted(os.listdir(input_path))  # Ensure files are sorted correctly
    
    for i in range(int(len(files) / 2)):
        top = np.load(os.path.join(input_path, f'array_{2 * i}.npy')).T
        bottom = np.load(os.path.join(input_path, f'array_{2 * i + 1}.npy')).T
        
        top = resize(top, (1588, 1588), order=0, mode='edge', anti_aliasing=False)
        bottom = resize(bottom, (1588, 1588), order=0, mode='edge', anti_aliasing=False)
        
        # Stack images vertically
        stitched_img = np.vstack((bottom, top))
        
        final_img = stitched_img[:3174, :1537]
        
        np.save(os.path.join(output_path, f'array_{i}.npy'), final_img)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and stitch numpy arrays.")
    parser.add_argument("input_path", type=str, help="Path to the folder containing input numpy files.")
    parser.add_argument("output_path", type=str, help="Path to save the processed numpy files.")
    
    args = parser.parse_args()
    process_and_stitch(args.input_path, args.output_path)
