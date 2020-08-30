#libraries
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

#generating a single csv file with data from all the xmls generated for every image
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text.strip().replace(" ", "",),
                     float(root.find('size')[0].text.strip().replace(" ", "")),
                     float(root.find('size')[1].text.strip().replace(" ", "")),
                     member[0].text.strip().replace(" ", ""),
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    #print(xml_list)
    return xml_df


def main():
    #for each directory named test and train
    for directory in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

#call main
main()