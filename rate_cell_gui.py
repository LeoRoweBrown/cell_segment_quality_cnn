import tkinter as tk
import tkinter.filedialog, tkinter.simpledialog
import json

from PIL import Image, ImageTk
import os
from glob import glob

class RateSegmentationGui():
    def __init__(self, root):
        self.root_dir = os.getcwd() 
        self.root = root
        self.main_canvas = tk.Canvas(root, width=800, height=600)
        self.main_canvas.pack()
        self.main_canvas.pack_propagate(False)
        self.control_frame = tk.Frame(self.main_canvas)
        self.control_frame.pack()

        self.cell_location_str_var = tk.StringVar()
        self.load_cells_button = tk.Button(self.control_frame, text="Browse cell segments location", command=self.browse_cell_dataset)
        self.cell_location_entry = tk.Entry(self.control_frame, textvariable=self.cell_location_str_var)
        self.cell_location_entry.grid(row=1, column=1)
        self.load_cells_button.grid(row=1, column=2)

        self.save_location_str_var = tk.StringVar()
        self.savefile_browser_button = tk.Button(self.control_frame, text="Browse ratings save location", 
                                                 command=self.browse_save_loc)
        self.save_location_entry = tk.Entry(self.control_frame, textvariable=self.save_location_str_var)
        self.save_location_entry.grid(row=2, column=1)
        self.savefile_browser_button.grid(row=2, column=2)
        self.save_file_button = tk.Button(self.control_frame, text="Save", 
                                                 command=self.save_ratings_to_file)
        self.save_file_button.grid(row=2, column=3, padx=2)
        self.restart_button= tk.Button(self.control_frame, text="Restart", 
                                                 command=self.reset_ratings)
        self.restart_button.grid(row=2, column=4, padx=2)

        self.rate_control = tk.Frame(self.control_frame)
        self.rate_control.grid(row=3, column=1, columnspan=2, padx=10, pady=10)
        self.good_button = tk.Button(self.rate_control, text="Good", command=lambda: self.assign_rating(2), bg='green')
        self.bad_button = tk.Button(self.rate_control, text="Bad", command=lambda: self.assign_rating(0), bg='red')
        self.unsure_button = tk.Button(self.rate_control, text="Unsure", command=lambda: self.assign_rating(1))
        self.good_button.pack(side=tk.LEFT)
        self.bad_button.pack(side=tk.LEFT)
        self.unsure_button.pack(side=tk.LEFT, padx=2)

        self.image_canvas = tk.Canvas(self.main_canvas)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.rating_data = []

    def assign_rating(self, rating):
        self.rating_data.append({
            'index': self.current_index,
            'filename': self.current_filename,
            'rating': rating
            })
        if self.current_index + 1 >= len(self.index_list):
            answer = tk.messagebox.askyesno("Done rating!", "You are done rating, save the rating to file?")
            if answer:
                self.save_ratings_to_file()
            else:
                self.reset_ratings()
        else:
            self.next_image()
    
    def save_ratings_to_file(self):
        filename = self.save_location_str_var.get()
        with open(filename, 'w') as f:
            json.dump(self.rating_data, f, indent=4)
    
    def browse_save_loc(self):
        filename = tk.filedialog.asksaveasfilename(defaultextension='.json', 
                                                   initialfile = os.path.join(self.root_dir, 'ratings'),
                                                   title = "Set rating savename")
        # Change entry contents
        self.save_location_str_var.set(filename)

    def browse_cell_dataset(self):
        dirname = tk.filedialog.askdirectory(initialdir = os.path.join(self.root_dir),
                                          title = "Select a dataset dir")
     
        # Change entry contents
        self.cell_location_str_var.set(dirname)
        print("got dir", dirname)
        print(self.cell_location_str_var.get())

        print("Loading new dataset")
        self.load_dataset()

    def next_image(self):
        i = self.current_index + 1
        self.set_index_and_image(i)
        print("index", i, "file", self.current_filename)

    def load_dataset(self):
        path = self.cell_location_str_var.get()
        print("searching...", os.path.join(path, '*.png'))
        self.image_path_list = glob(os.path.join(path, '*.png'))
        self.filename_list = [os.path.basename(path) for path in self.image_path_list]
        self.index_list = range(len(self.image_path_list))
        self.image_list = []
        print("path list", self.image_path_list)
        for image in self.image_path_list:
            img = Image.open(image, 'r')
            self.image_list.append(img)
        self.save_location_str_var.set("")
        print("PIL images", self.image_list)
        self.set_index_and_image(0)
    
    def set_index_and_image(self, i):
        self.current_index = i
        self.current_pil_img = ImageTk.PhotoImage(self.image_list[i])
        self.current_cell_image = self.image_canvas.create_image(0, 0, image=self.current_pil_img, anchor='nw')
        self.current_filename = self.filename_list[i]
    
    def reset_ratings(self):
        self.set_index_and_image(0)
        self.rating_data = []
    
def main():
    root = tk.Tk()
    gui = RateSegmentationGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()