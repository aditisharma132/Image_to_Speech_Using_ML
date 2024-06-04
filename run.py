import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
import pyttsx3
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM
import io

class ImageToSpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to Speech Converter")

        # URL entry
        self.url_label = tk.Label(root, text="Enter Image URL:")
        self.url_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.url_entry = tk.Entry(root, width=50)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5)

        # Buttons
        self.url_button = tk.Button(root, text="Process URL", command=self.process_url)
        self.url_button.grid(row=1, column=0, padx=5, pady=5)

        self.camera_button = tk.Button(root, text="Open Camera", command=self.open_camera)
        self.camera_button.grid(row=1, column=1, padx=5, pady=5)

        # Speech output
        self.speech_label = tk.Label(root, text="Speech Output:")
        self.speech_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.speech_text = tk.Text(root, width=50, height=5)
        self.speech_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

        # Initialize camera
        self.cap = None

    def process_url(self):
        url = self.url_entry.get()
        self.process_image_url(url)

    def process_image_url(self, url):
        try:
            # Open image from URL
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))

            # Display image
            self.display_image(image)

            # Generate speech
            self.generate_speech_from_image(image)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def open_camera(self):
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)

            # Process camera feed
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Display camera feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                self.display_image(image)

                # Generate speech
                self.generate_speech_from_image(image)

                # Update GUI
                self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

        finally:
            # Release camera
            if self.cap:
                self.cap.release()

    def display_image(self, image):
        # Resize and display image
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def generate_speech_from_image(self, image):
        try:
            # Preprocess image
            image = image.resize((224, 224))
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

            # Generate caption
            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Update speech output
            self.speech_text.delete(1.0, tk.END)
            self.speech_text.insert(tk.END, generated_caption)

            # Convert text to speech
            engine = pyttsx3.init()
            engine.say(generated_caption)
            engine.runAndWait()

        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToSpeechApp(root)
    root.mainloop()
