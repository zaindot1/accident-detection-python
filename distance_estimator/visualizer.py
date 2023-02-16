
import glob
import os
import argparse
import cv2
import numpy as np
import pandas as pd

def visualizer():
    def draw_text(img, text,
                  font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  pos=(0, 0),
                  font_scale=1,
                  font_thickness=1,
                  text_color=(255, 255, 255),
                  text_color_bg=(0, 0, 0)
                  ):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w + 2, y + text_h + 2), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

        return text_size

    def write_predictions_on_frames(df):
        for idx, row in df.iterrows():
            fn = "{}.png".format(int(row['frame']))
            fp = os.path.join(os.getcwd(), fn)
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            if os.path.exists(fp):
                im = cv2.imread(fp)
                string = "{} meters".format(row['distance'])
                fontface = cv2.FONT_HERSHEY_COMPLEX_SMALL
                draw_text(im, string, fontface, (x1 + 3, y1 + 3));
                cv2.imwrite(fn, im)
                cv2.waitKey(0)
            else:
                print(fp)

    # parse arguments
    csvfile_path = '../distance_estimator/predictions.csv'
    frames_dir = '../object_detector/results/frames/'
    fps = '30'
    results_dir = '.'

    # write predictions on frames
    df = pd.read_csv(csvfile_path)
    os.chdir(frames_dir)
    write_predictions_on_frames(df)

    os.chdir(results_dir)


