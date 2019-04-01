import labelbox.exporters.voc_exporter as lb2pa
import json

labeled_data = "export-2019-04-01t09_19_46.617z.json" # file path to JSON export with XY paths

# where to write VOC annotation and image outputs
annotations_output_dir = "Annotations"
images_output_dir = "JPEGImages"

lb2pa.from_json(labeled_data, annotations_output_dir, images_output_dir, label_format='XY')
