import os
import tkinter as tk
from PIL import Image, ImageTk


def extract_number(f):
    basename = os.path.basename(f)
    name, _ = os.path.splitext(basename)
    parts = name.split('_')
    try:
        num = int(parts[-1])
    except ValueError:
        num = 0
    return num


class ImageClassifier:
    def __init__(self, image_folder, start_index=None, end_index=None, local_prefix=None, server_prefix=None):
        self.current_image = None
        self.classified_set = None
        self.image_folder = image_folder
        self.local_prefix = local_prefix
        self.server_prefix = server_prefix
        self.class_a_file = 'class_aligned.txt'
        self.class_b_file = 'class_not_aligned.txt'
        self.image_files = self.load_images()
        self.classified_images = []
        self.current_index = 0
        self.load_classified_images()
        self.filter_unclassified_images()

        self.start_index = start_index
        self.end_index = end_index
        self.filter_by_index_range()

        self.root = tk.Tk()
        self.panel = tk.Label(self.root)
        self.panel.pack()
        self.emoji_label = None

    def load_images(self):
        image_files = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

        image_files.sort(key=extract_number, reverse=True)
        return image_files

    def load_classified_images(self):
        self.classified_set = set()
        for filename in [self.class_a_file, self.class_b_file]:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    for line in f:
                        server_path = line.strip()
                        # Map server path back to local path
                        if self.local_prefix and self.server_prefix:
                            if server_path.startswith(self.server_prefix):
                                relative_path = os.path.relpath(
                                    server_path, self.server_prefix)
                                local_path = os.path.join(
                                    self.local_prefix, relative_path)
                                local_path = os.path.normpath(local_path)
                            else:
                                print(f"Warning: Path '{server_path}' does not "
                                      f"start with server prefix '{self.server_prefix}'. "
                                      "Using original path.")
                                local_path = server_path
                        else:
                            local_path = server_path
                        self.classified_set.add(local_path)

    def filter_unclassified_images(self):
        self.image_files = [
            f for f in self.image_files if f not in self.classified_set
        ]

    def filter_by_index_range(self):
        if self.start_index is not None or self.end_index is not None:
            filtered_files = []
            for f in self.image_files:
                num = extract_number(f)
                if self.start_index is not None and num < self.start_index:
                    continue
                if self.end_index is not None and num > self.end_index:
                    continue
                filtered_files.append(f)
            self.image_files = filtered_files

    def run(self):
        if not self.image_files:
            print("All images have been classified!")
            self.root.destroy()
            return
        self.show_image()
        self.root.mainloop()

    def show_image(self):
        if self.current_index >= len(self.image_files):
            print("All images have been classified!")
            self.root.destroy()
            return

        image_path = self.image_files[self.current_index]
        try:
            img = Image.open(image_path)
            img = img.resize((800, 600))
            self.current_image = img
            photo = ImageTk.PhotoImage(img)
            self.panel.config(image=photo)
            self.panel.image = photo
            self.root.title(
                f"Image {image_path}. {self.current_index + 1} of {len(self.image_files)}"
            )
            self.root.bind('<Key>', self.on_key)
        except Exception as e:
            print(f"Failed to open {image_path}: {e}")
            self.current_index += 1
            self.show_image()

    def on_key(self, event):
        if event.keysym == 'Right':
            self.classify_image('a')
        elif event.keysym == 'Left':
            self.classify_image('b')
        elif event.keysym == 'u':
            self.undo_classification()
        elif event.keysym == 'Escape':
            self.root.destroy()

    def classify_image(self, classification):
        image_path = self.image_files[self.current_index]
        self.classified_images.append((image_path, classification))
        self.update_file(image_path, classification)
        self.show_emoji(classification)
        self.root.after(200, self.next_image)  # for presentation
        # self.root.after(0, self.next_image)  # for efficiency

    def update_file(self, image_path, classification):
        # Replace local_prefix with server_prefix in the image path
        if self.local_prefix and self.server_prefix:
            if image_path.startswith(self.local_prefix):
                relative_path = os.path.relpath(image_path, self.local_prefix)
                server_path = os.path.join(self.server_prefix, relative_path)
                server_path = server_path.replace('\\', '/')  # Ensure Unix-style paths
            else:
                print(
                    f"Warning: Image path '{image_path}' does not start with local prefix '{self.local_prefix}'. Using original path.")
                server_path = image_path
        else:
            server_path = image_path

        filename = self.class_a_file if classification == 'a' else self.class_b_file
        with open(filename, 'a') as f:
            f.write(server_path + '\n')
            f.flush()

    def show_emoji(self, classification):
        if classification == 'a':
            emoji = '✅'
        elif classification == 'b':
            emoji = '❌'
        elif classification == 'undo':
            emoji = '⬅️'
        else:
            emoji = ''
        if self.emoji_label:
            self.emoji_label.destroy()
        self.emoji_label = tk.Label(
            self.root, text=emoji, font=('Arial', 100), bg='white'
        )
        self.emoji_label.place(relx=0.5, rely=0.5, anchor='center')
        self.root.update()

    def next_image(self):
        if self.emoji_label:
            self.emoji_label.destroy()
            self.emoji_label = None
        self.current_index += 1
        self.show_image()

    def undo_classification(self):
        if self.current_index == 0 or not self.classified_images:
            print("No classification to undo.")
            return
        last_image, last_class = self.classified_images.pop()
        self.current_index -= 1
        self.remove_last_entry(last_class)
        self.show_emoji('undo')
        self.root.after(200, self.show_previous_image)  # for presentation

    def show_previous_image(self):
        # Destroy the emoji label before showing the image
        if self.emoji_label:
            self.emoji_label.destroy()
            self.emoji_label = None
        self.show_image()

    def remove_last_entry(self, classification):
        filename = self.class_a_file if classification == 'a' else self.class_b_file
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
            if lines:
                with open(filename, 'w') as f:
                    f.writelines(lines[:-1])
                    f.flush()


def main():
    image_folder = '/Users/nullrequest/Documents/USI/neuralwave_hackaton/data/train_set/good_light'

    local_prefix = '/Users/nullrequest/Documents/USI/neuralwave_hackaton/data'
    server_prefix = ('/teamspace/s3_connections/dtp-sbm-segmentation-video-tasks-bars-stopper-alignment-images'
                     '-hackaton-usi')

    classifier = ImageClassifier(
        image_folder,
        local_prefix=local_prefix,
        server_prefix=server_prefix
    )

    classifier.run()


if __name__ == '__main__':
    main()
