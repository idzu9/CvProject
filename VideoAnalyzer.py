#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
import matplotlib

from huggingface_hub import login
from huggingface_hub import snapshot_download

from transformers import Sam3Processor, Sam3Model
import requests

# download the model

# SAM3 git repo with isntallation details
# https://github.com/facebookresearch/sam3/tree/main

# arrow buttons key mappings for different platforms
UP_KEYS = {2490368, 65362, 82, 63232, 0}
DOWN_KEYS = {2621440, 65364, 84, 63233, 1}
LEFT_KEYS = {2424832, 65361, 81, 63234, 2}
RIGHT_KEYS = {2555904, 65363, 83, 63235, 3}
ENTER_KEYS = {13}
BACKSPACE_KEYS = {8, 127}
PRINTABLE_ASCII_RANGE = range(32, 127)

class OpenCVVideoUI:

	def __init__(self, model, processor, device):    
		# file selector dialog, closing file dialog window will close the app
		self.video_path = self.select_video_file()
		if not self.video_path:
			return

		self.cap = cv2.VideoCapture(self.video_path)
		self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		# get frame rate of the video and initialize delay
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		if self.total_frames <= 0:
			self.total_frames = 100
		self.fps = self.fps if self.fps > 0 else 30
		self.delay = int(1000.0 / self.fps)

		self.current_frame_idx = 0
		self.paused = True
		self.current_frame = None

		# Interaction & Analytics State
		self.clicks = []          # Stores (x, y, label)
		self.text_input_mode = False
		self.text_prompt = ""     # Live text typed for SAM

		# Setup OpenCV Window & Mouse Callback
		self.window_name = "OpenCV SAM Video Processor"
		cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
		cv2.setMouseCallback(self.window_name, self.mouse_callback)

		# Create Timeline Trackbar
		cv2.createTrackbar("Timeline", self.window_name, 0, self.total_frames - 1, self.on_trackbar)

		# # SAM3 part
		# self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.model = model
		self.processor = processor
		self.device = device
		# print(f"DEVICE IS: {self.device}")
		# self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
		# self.processor = Sam3Processor.from_pretrained("facebook/sam3")

		# print('Parameters: %.0fM' % (self.model.num_parameters() / 1e6))

		return

	# select a video file from the file system
	def select_video_file(self):
		root = tk.Tk()
		root.withdraw()
		file_path = filedialog.askopenfilename(
			title="Select Video File",
			filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All Files", "*.*")]
		)
		return file_path

	# trackbar updates
	def on_trackbar(self, val):
		if not self.paused:
			return

		self.current_frame_idx = val
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
		ret, frame = self.cap.read()
		if ret:
			self.current_frame = frame
			self.refresh_display()

	# mouse callbacks
	def mouse_callback(self, event, x, y, flags, param):
		if self.paused and not self.text_input_mode:
			if event == cv2.EVENT_LBUTTONDOWN:
				self.clicks.append((x, y, 1)) # Foreground point
				print(f"[Point Prompt] Added Positive at ({x}, {y})")
				self.refresh_display()
			elif event == cv2.EVENT_RBUTTONDOWN:
				self.clicks.append((x, y, 0)) # Background point
				print(f"[Point Prompt] Added Negative at ({x}, {y})")
				self.refresh_display()

	def draw_status_bar(self, display_img):
		h, w, _ = display_img.shape

		cv2.rectangle(display_img, (0, 0), (w, 50), (30, 30, 30), -1)
		box_color = (0, 100, 255) if self.text_input_mode else (50, 50, 50)
		paying_state = 'PLAYING' if not self.paused else 'PAUSED'
		status = f"Frame: {self.current_frame_idx}/{self.total_frames} | {paying_state :<7}"
		cv2.putText(display_img, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

	# refresh display
	def refresh_display(self):
		if self.current_frame is None:
			return

		display_img = self.current_frame.copy()
		h, w, _ = display_img.shape

		# 1. Draw point prompts
		for (x, y, label) in self.clicks:
			color = (0, 255, 0) if label == 1 else (0, 0, 255)
			cv2.circle(display_img, (x, y), 6, color, -1)
			cv2.circle(display_img, (x, y), 7, (255, 255, 255), 1)

		# 2. Draw Top Status Banner
		self.draw_status_bar(display_img)

		# 3. Draw In-Window Text Box Banner at the Bottom
		box_color = (0, 100, 255) if self.text_input_mode else (50, 50, 50)
		cv2.rectangle(display_img, (10, h - 60), (w - 10, h - 10), box_color, -1)
		
		if self.text_input_mode:
			prompt_display = f"SAM Text Prompt (Type & Enter): {self.text_prompt}_"
		else:
			prompt_display = f"SAM Text Prompt: '{self.text_prompt}' (Press 't' to edit)"
			
		cv2.putText(display_img, prompt_display, (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Update trackbar position
		cv2.setTrackbarPos("Timeline", self.window_name, self.current_frame_idx)
		cv2.imshow(self.window_name, display_img)

	def run(self):
		if not self.cap or not self.cap.isOpened():
			return False

		# Read first frame
		ret, frame = self.cap.read()
		if ret:
			self.current_frame = frame
			self.refresh_display()

		inputs = self.processor(images=self.current_frame, text="car", return_tensors="pt").to(self.device)
		with torch.no_grad():
			outputs = self.model(**inputs)

		results = self.processor.post_process_instance_segmentation(
			outputs,
			threshold=0.5,
			mask_threshold=0.5,
			target_sizes=inputs.get("original_sizes").tolist()
		)[0]

		print(f"Found {len(results['masks'])} objects")

		results = self.processor.post_process_instance_segmentation(
			outputs,
			threshold=0.5,
			mask_threshold=0.5,
			target_sizes=inputs.get("original_sizes").tolist()
		)[0]

		masks = results["masks"]
		masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
		n_masks = masks.shape[0]
		cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
		colors = [
			tuple(int(c * 255) for c in cmap(i)[:3])
			for i in range(n_masks)
		]

		for mask, color in zip(masks, colors):
			# Ensure mask is binary (0 or 1)
			binary_mask = (mask > 0).astype(np.uint8)

			# Create a colored layer of the same size as the image
			colored_mask = np.zeros_like(self.current_frame, dtype=np.uint8)
			colored_mask[:] = color

			# Apply mask to the colored layer
			colored_mask = cv2.bitwise_and(
				colored_mask, colored_mask, mask=binary_mask * 255
			)

			# Blend the colored mask onto the image with transparency (e.g., 50% opacity)
			# alpha = 0.5, beta = 1.0, gamma = 0.0
			self.current_frame = cv2.addWeighted(self.current_frame, 1.0, colored_mask, 0.5, 0)

		self.refresh_display()

		print("\n--- CONTROLS ---")
		print("[SPACEBAR] : Toggle Play / Pause")
		print("[ RIGHT ARROW ] : Step 1 frame forward")
		print("[ LEFT ARROW ]  : Step 1 frame backward")
		print("[ T ]           : Activate In-Window Text Prompt Typing Mode")
		print("[ S ]           : Run SAM Analysis on current frame")
		print("[ C ]           : Clear prompts")
		print("[ Q ]           : Quit\n")

		should_open_new = False
		while self.cap.isOpened():
			try:
				if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
					print("Window closed by user. Opening file selector...")
					should_open_new = True
					break
			except cv2.error:
				should_open_new = True
				break

			# Only read automatically if playing
			if not self.paused:
				if not self.read_next_frame():
					continue

			wait_time = 0 if self.paused else max(1, self.delay)
			# reading key input
			key = cv2.waitKeyEx(wait_time)

			if key == ord('q') or key == ord('Q'):
				should_open_new = False
				break

			# --- TEXT TYPING MODE HANDLER ---
			if self.text_input_mode:
				# Enter key saves text prompt
				if key in ENTER_KEYS:
					self.text_input_mode = False
					print(f"-> Text Prompt Confirmed: '{self.text_prompt}'")
					self.refresh_display()
				# Backspace
				elif key in BACKSPACE_KEYS:
					self.text_prompt = self.text_prompt[:-1]
					self.refresh_display()
				# printable ASCII characters
				elif key in PRINTABLE_ASCII_RANGE:
					self.text_prompt += chr(key)
					self.refresh_display()
				continue # Skip standard shortcuts while typing text

			self.handle_standard_controls(key)

		self.cap.release()
		cv2.destroyAllWindows()
		return should_open_new

	# step frame in direction
	def step_frames(self, direction):
		self.set_pause_state(True)

		new_idx = self.current_frame_idx + direction
		self.current_frame_idx = max(0, min(self.total_frames - 1, new_idx))
				
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
		ret, frame = self.cap.read()
		if ret:
			self.current_frame = frame
			self.refresh_display()

	def set_pause_state(self, new_state):
		self.paused = new_state
		self.refresh_display()
	
	def toggle_pause_state(self):
		self.paused = not self.paused
		self.refresh_display()

	def handle_standard_controls(self, key):
		# play and pasue
		if key == ord(' '):
			self.toggle_pause_state()
		# step forward 1 frame
		elif key in RIGHT_KEYS: 
			self.step_frames(1)
		 # step backward 1 frame
		elif key in LEFT_KEYS:
			self.step_frames(-1)
		# toggle text typing mode for SAM
		elif key == ord('t') or key == ord('T'):
			if self.paused:
				self.text_input_mode = True
				print("\n[Text Mode Active] Type your prompt on your keyboard. Press ENTER when done.")
				self.refresh_display()
		# run analysis
		elif (key == ord('s') or key == ord('S')) and self.paused: 
			print(f"\n[Running SAM Analysis]")
			print(f" -> Frame: {self.current_frame_idx}")
			print(f" -> Text Prompt: '{self.text_prompt}'")
			print(f" -> Point Prompts: {self.clicks}")
		# clear points
		elif key == ord('c') or key == ord('C'): 
			self.clicks.clear()
			self.text_prompt = ""
			print("Cleared all prompts.")
			self.refresh_display()

	def read_next_frame(self):
		ret, frame = self.cap.read()
		if not ret:
			self.paused = True
			return False

		self.current_frame = frame
		self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
		self.refresh_display()
		return True

def main():
	print("\n\nGreetings dear user, before using the following tool you need the following:\n")
	print("1. Hugging face account\n")
	print("2. Get access to the https://huggingface.co/facebook/sam3 repo\n")
	print("3. You can authorize to Hugging face via access token or browser\n")

	# login()

	# local_model_path = snapshot_download(
	# 	repo_id="facebook/sam3", 
	# 	local_dir="./sam3_local_weights" # Saves it in a folder next to your script
	# )

	# print(f"\nModel downloaded to: {local_model_path}")

	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True
	torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

	# SAM3 part
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	print(f"DEVICE IS: {device}")
	model = Sam3Model.from_pretrained("facebook/sam3").to(device)
	processor = Sam3Processor.from_pretrained("facebook/sam3")
	print('Parameters: %.0fM' % (model.num_parameters() / 1e6))

	while True:
		app = OpenCVVideoUI(model, processor, device)
		if not hasattr(app, 'cap') or app.cap is None:
			break
			
		open_another = app.run()
		if not open_another:
			break    

if __name__ == "__main__":    
	main()