# importing the module
import cv2

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import argparse
import os.path as osp
import pickle

def create_circle(x, y, r, canvas): #center coordinates, radius
    return canvas.create_oval(x - r, y - r, x + r, y + r, fill='red')

def get_diameter(points_i_px, points_f_px):
    points_i_m = (points_i_px - origin_px) * m2px_scale
    points_f_m = (points_f_px - origin_px) * m2px_scale

    return abs(np.linalg.norm(np.array(points_i_m) - np.array(points_f_m)))

def next_limb_prompt():
    if canvas.index < len(canvas.limbs):
        print(f'Measure diameter of the {canvas.limbs[canvas.index].upper()}')

    else:
        print(f'MEASUREMENTS DONE!')# print(canvas.body_info)
        # f = open('')
        filename = 'body_info.pkl'
        with open(osp.join(args.subject_dir, filename),'wb') as f:
            pickle.dump(canvas.body_info, f)
#

def draw_line(event):
	# print(event.type)
    if canvas.index < len(canvas.limbs):
        if str(event.type) == 'ButtonPress':
            canvas.old_coords = event.x, event.y
            create_circle(event.x, event.y, 2, canvas)

        elif str(event.type) == 'Motion':
            x, y = event.x, event.y
            x1, y1 = canvas.old_coords
            canvas.delete(canvas.old_line_id)
            canvas.old_line_id = canvas.create_line(x, y, x1, y1)

        elif str(event.type) == 'ButtonRelease':
            canvas.old_line_id = None
            create_circle(event.x, event.y, 2, canvas)
            
            diameter = get_diameter(canvas.old_coords, (event.x, event.y))
            print(diameter)
            radius = diameter/2
            # canvas.body_info[canvas.limbs[canvas.index]] = [canvas.old_coords, (event.x, event.y)]
            canvas.body_info[canvas.limbs[canvas.index]][1] = radius
            canvas.index += 1
            next_limb_prompt()

def reset_coords(event):
    canvas.old_coords = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', type=str, default='TEST')
    parser.add_argument('--pose-dir', type=str, default='TEST')
    args = parser.parse_args()

    with open(osp.join(args.pose_dir, 'sim_origin_data.pkl'),'rb') as f:
        data = pickle.load(f)
        dist = data['dist']
        mtx = data['mtx']
        centers_px = data['centers_px']
        centers_m = data['centers_m']
        origin_px = data['origin_px']
        origin_m = data['origin_m']
        m2px_scale = data['m2px_scale']

    root = tk.Tk() 

    scale = 1
    canvas = tk.Canvas(root, width=1280*scale, height=720*scale)
    canvas.pack()


    bg= ImageTk.PhotoImage(file=osp.join(args.pose_dir, 'uncovered_rgb.png'))
    # image = Image.open(osp.join(args.save_dir, 'pose_0/uncovered_rgb.png'))
    # resized = image.resize((int(1280*scale), int(720*scale)), Image.ANTIALIAS)
    # bg = ImageTk.PhotoImage(resized)

    canvas.create_image(0,0,image=bg, anchor="nw")
    canvas.old_coords = None
    canvas.old_line_id = None
    canvas.body_info = {
        'head':[None,0],        #[length, radius] - don't need to collect length
        'upperchest':[None,0],  
        'waist':[None,0], 
        'upperarm':[None,0], 
        'forearm':[None,0],
        'hand':[None,0],
        'thigh':[None,0], 
        'shin':[None,0], 
        'foot':[None,0]
    }
    canvas.limbs = [
        'head','upperchest','waist', 
        'upperarm','forearm','hand',
        'thigh','shin','foot'
    ]
    canvas.index = 0

    root.bind('<ButtonPress-1>', draw_line)
    root.bind('<ButtonRelease-1>', draw_line)
    root.bind('<B1-Motion>', draw_line)

    next_limb_prompt()

    #root.bind('<B1-Motion>', draw)
    #root.bind('<ButtonRelease-1>', reset_coords)

    root.mainloop()

