import os


def collect_file_paths(data_path):
    real_path = os.path.join(data_path, 'train_real')
    label_path = os.path.join(data_path, 'train_label')

    real_files = collect_image_files(real_path)
    label_files = collect_image_files(label_path)

    matched_files = match_files(real_files, label_files, label_path)

    if not matched_files:
        return None, None

    return zip(*matched_files)


def collect_image_files(data_path):
    image_files = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".png"):
                file_path = os.path.join(root, file)
                image_files.append(file_path)

    return image_files


def match_files(real_files, label_files, label_path):
    matched_files = []
    for real_file in real_files:
        filename = os.path.splitext(os.path.basename(real_file))[0]
        label_filename = f"{filename}_label"

        # Check if the label filename (without extension) is present in label_files
        matching_label_files = [file for file in label_files if os.path.splitext(os.path.basename(file))[0] == label_filename]

        if matching_label_files:
            label_file = matching_label_files[0]
            matched_files.append((real_file, label_file))

    return matched_files
