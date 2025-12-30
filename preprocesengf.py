import os
import shutil


import os

# 1. Define the base directory for the original dataset
original_base_dir = './robocon-vision/data/'

# 2. Construct the full paths to the original 'Image' and 'Annotations' directories
original_image_dir = os.path.join(original_base_dir, 'Image')
original_annotations_dir = os.path.join(original_base_dir, 'Annotations')

paired_files = []

# 3. Get a list of all image filenames (without their paths) from the 'Image' directory
image_files = [f for f in os.listdir(original_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    # Extract filename without extension
    img_name_without_ext = os.path.splitext(img_file)[0]

    # 4. Construct the corresponding annotation filename (assuming XML annotations)
    annotation_file = img_name_without_ext + '.xml'

    # Construct full paths
    full_image_path = os.path.join(original_image_dir, img_file)
    full_annotation_path = os.path.join(original_annotations_dir, annotation_file)

    # 5. Ensure both files exist before adding them to the list
    if os.path.exists(full_image_path) and os.path.exists(full_annotation_path):
        paired_files.append((full_image_path, full_annotation_path))

print(f"Found {len(paired_files)} paired image and annotation files.")
# Display the first 5 paired files as an example
print("Example paired files:")
for i, (img_path, ann_path) in enumerate(paired_files[:5]):
    print(f"  Image: {img_path}\n  Annotation: {ann_path}")
    if i == 4:
        break


root = './data'
folders = os.listdir(root)
f = ['Annotations','Image']
for _ in f:
    for i in range(len(folders-1)):
        root_file = os.path.join(root, folders[i], _)
        all_files = os.listdir(root_file)
        name = []
        cnt = 0
        for file in all_files:
            if _ == 'Annotations':
                name.append(os.path.splitext(file)[0])
            else:
                if name == os.path.splitext(file)[0]:
                    cnt+=1

        items = len(all_files)
        train_f = items//10 * 6 + 1
        val_f = items//10 * 3 + 1

        train_i = all_files[:train_f-1]
        val_i = all_files[train_f:train_f+val_f-1]
        test_i = all_files[train_f+val_f:]

        for file in train_i:
            shutil.move(os.paht.join(root_file,file), os.path.join(root, 'train',file))

        for file in val_i:
            shutil.move(os.paht.join(root_file,file), os.path.join(root, 'val', file))

        for file in test_i:
            shutil.move(os.paht.join(root_file,file), os.path.join(root, 'test',file))

        print(f"pair:{cnt}")
