import os

scene_name = 'large_corridor_25'
source_folder = f'data/messy_rooms/{scene_name}/color'
# destination_depth_folder = f'data/messy_rooms/{scene_name}/depths'
destination_rgb_folder = f'data/messy_rooms/{scene_name}/input'


# if not os.path.exists(destination_depth_folder):
#     os.makedirs(destination_depth_folder)

if not os.path.exists(destination_rgb_folder):
    os.makedirs(destination_rgb_folder)


# select_indices = [f"depth{i:06d}.png" for i in range(0, 2000, 10)]
# select_indices = [f"frame{i:06d}.jpg" for i in range(0, 2000, 10)]
image_names = os.listdir(source_folder)
image_names.sort()
select_image_names = [image_names[i] for i in range(0, 600, 6)]
count = 0

for select_image_name in select_image_names:
    source_file_path = os.path.join(source_folder, select_image_name)
    destination_file_path = os.path.join(destination_rgb_folder, select_image_name)  #destination_depth_folder

    from shutil import copy
    copy(source_file_path, destination_file_path)
    count += 1

print("selection completed and copied")
print(count)