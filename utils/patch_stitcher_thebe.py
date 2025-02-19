import os



import numpy as np

from skimage.transform import resize




thebe_path = '/home/jorge/GhassanGT Dropbox/Jorge Quesada/Jorge/InSync/BIG and DATA/thebe_data/test/patches_processed/labels/'

out_path =  '/home/jorge/GhassanGT Dropbox/Jorge Quesada/Jorge/InSync/BIG and DATA/thebe_data/test/patches_processed/labels_restitched/'


if not os.path.exists(out_path):
    os.makedirs(out_path)

files = os.listdir(thebe_path)

for i in range(int(len(files)/2)):



    top = np.load(thebe_path + 'array_' +  str(2*i) + '.npy')

    bottom = np.load(thebe_path + 'array_' + str(2*i+1) + '.npy')



    top = resize(top, (1588,1588), order=0, mode='edge', anti_aliasing=False)

    bottom = resize(bottom, (1588,1588), order=0, mode='edge', anti_aliasing=False)



    # Stack images vertically

    stitched_img = np.vstack((bottom, top))



    final_img = stitched_img[:3174, :1537]



    np.save(out_path + 'array_' + str(i) + '.npy', final_img)