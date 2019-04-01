import voc_exporter as ve

labeled_data = "original_dataset/smokes.json" # file path to JSON export with XY paths

# where to write VOC annotation and image outputs
annotations_output_dir = "original_dataset/Annotations"
images_output_dir = "original_dataset/JPEGImages"

ve.from_json(labeled_data, annotations_output_dir, images_output_dir, label_format='XY')
