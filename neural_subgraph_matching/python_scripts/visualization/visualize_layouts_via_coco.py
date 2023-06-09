import pycocotools
import cv2
import os

from pycocotools.coco import COCO
import layoutparser as lp
import random
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# adapted from: https://layout-parser.readthedocs.io/en/latest/example/load_coco/index.html#loading-and-visualizing-layouts-using-layout-parser

def load_coco_annotations(annotations, coco=None):
    """
    Args:
        annotations (List):
            a list of coco annotaions for the current image
        coco (`optional`, defaults to `False`):
            COCO annotation object instance. If set, this function will
            convert the loaded annotation category ids to category names
            set in COCO.categories
    """

    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele['bbox']

        layout.append(
            lp.TextBlock(
                block = lp.Rectangle(x, y, w+x, h+y),
                type  = ele['category_id'] if coco is None else coco.cats[ele['category_id']]['name'],
                id = ele['id']
            )
        )

    return layout


file_path = os.getcwd()

coco_anno_path = os.path.join(file_path, "jsons/62_0.json")

coco_img_path = os.path.join(file_path, "images/62_0")

save_path = os.path.join(file_path, "annotations/test.txt")

#def random_layout_viz(N, coco_anno_path, coco_img_path, save_path, detailed=False):
    
def random_layout_viz(N, detailed=False):    
    """
    Args:
        N (integer):
            Number of layouts to visualize at random
        coco_anno_path (string):
            Path to COCO annotations file
        coco_img_path (string):
            Path to image directory
        save_path (string):
            Path to save visualized layouts
        detailed (boolean):
            If `True`, outputs visualizations with extra information 
    """

    coco = COCO(coco_anno_path)

    pp = PdfPages(save_path)

    color_map = {
        'newspaper_header':         'red',
        'masthead':                 'blue',
        'article':                  'green',
        'headline':                 'purple',
        'photograph':               'pink',
        'cartoon_or_advertisement': 'orange',
        'image_caption':            'yellow',
        'page_number':              'black',
        'table':                    'gray'
    }
    N = 1
    for image_id in random.sample(coco.imgs.keys(), N):
        image_info = coco.imgs[image_id]
        annotations = coco.loadAnns(coco.getAnnIds([image_id]))

        image = cv2.imread(f'{coco_img_path}/{image_info["file_name"]}')
        layout = load_coco_annotations(annotations, coco)

        if detailed is False:
            viz = lp.draw_box(image, layout, color_map=color_map)
        else: 
            viz = lp.draw_box(image, [b.set(id=f'{b.id}/{b.type}') for b in layout],
                    color_map=color_map,
                    show_element_id=True, 
                    id_font_size=10,
                    id_text_background_color='grey',
                    id_text_color='white')
        
        plt.figure()
        plt.imshow(viz)
        plt.axis('off')
        pp.savefig(dpi=1000)

    pp.close()

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of layouts to visualize at random")
    parser.add_argument('-d', action='store_true', help="Output visualizations with extra information?")
    parser.add_argument("--coco_anno_path", help="Path to COCO annotations file")
    parser.add_argument("--coco_img_path", help="Path to image directory")
    parser.add_argument("--save_path", help="Path to save visualized layouts")
    args = parser.parse_args()

    random_layout_viz(
        N=args.n, 
        detailed=args.d) 
    
    # random_layout_viz(
    #     N=args.n, 
    #     coco_anno_path=args.coco_anno_path, 
    #     coco_img_path=args.coco_img_path, 
    #     save_path=args.save_path,
    #     detailed=args.d) 