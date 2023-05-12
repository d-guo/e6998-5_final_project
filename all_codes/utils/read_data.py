import xmltodict
import os

class DataConverter:
    
    def __init__(self):
        self.class_label = {}
        self.max_class_label = -1
    
    def convert_xml_data(self, folder_path="../scratch/svhn-voc-annotation-format/annotation/train/",
                                output_path="../data/google_digit_data/train/"):
        
        for file in os.listdir(folder_path):
            
            file_path = os.path.join(folder_path, file)
            self.write_to_txt(file_path, file, output_path)
    
    def write_to_txt(self, xml_file_path, xml_file_name, output_path):
        
        with open(xml_file_path, 'r') as f:
            data = f.read()
            f.close()
            
        data = xmltodict.parse(data, encoding='utf-8')['annotation']
        new_file_name = xml_file_name.split('.')[0] + '.txt'
        new_file_path = os.path.join(output_path, new_file_name)        

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if os.path.isfile(new_file_path):
            os.remove(new_file_path)
        
        # create an empty file
        with open(new_file_path, 'w') as fp:
            pass

        # set the datatype for files containing only one annotated object
        if isinstance(data['object'], dict):
            data['object'] = [data['object']]

        width = int(data['size']['width'])
        height = int(data['size']['height'])

        for obj in data['object']:
            if obj['name'] == 'bg':
                pass
            else:
                if obj['name'] not in self.class_label:
                    self.max_class_label += 1
                    self.class_label[obj['name']] = self.max_class_label
                class_label = self.class_label[obj['name']]

                with open(new_file_path, 'a') as f:
                    x_min = int(obj['bndbox']['xmin'])
                    x_max = int(obj['bndbox']['xmax'])
                    y_min = int(obj['bndbox']['ymin'])
                    y_max = int(obj['bndbox']['ymax'])
                    
                    box_x_center = (x_min+x_max)/(2*width)
                    box_y_center = (y_min+y_max)/(2*height)
                    box_width = (x_max-x_min)/width
                    box_height = (y_max-y_min)/height
                    
                    f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_label, box_x_center, box_y_center, box_width, box_height))
                    f.close()

    def remove_images(self, train_image_path, train_label_path, valid_image_path, valid_label_path,
                        test_image_path, test_label_path):
        # remove any image that did not output .txt file

        train_images_list = set([i for i in os.listdir(train_image_path) if 'jpg' in i])
        train_label_list = set([i for i in os.listdir(train_label_path) if 'txt' in i])

        valid_images_list = set([i for i in os.listdir(valid_image_path) if 'jpg' in i])
        valid_label_list = set([i for i in os.listdir(valid_label_path) if 'txt' in i])

        test_images_list = set([i for i in os.listdir(test_image_path) if 'jpg' in i])
        test_label_list = set([i for i in os.listdir(test_label_path) if 'txt' in i])

        for d in train_images_list-train_label_list:
            os.remove(os.path.join(train_images_list.replace('images_backup','images'), d))
        
        for d in set.intersection(valid_images_list,valid_label_list):
            os.remove(os.path.join(valid_images_list.replace('images_backup','images'), d))

        for d in set.intersection(test_images_list,test_label_list):
            os.remove(os.path.join(test_images_list.replace('images_backup','images'), d))