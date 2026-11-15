#!/usr/bin/env python

import VideoAnalyzer as va

def print_how_to():
	print("\n\nGreetings dear user, how to for the tool is the following:\n")
	print("\n1. Choose a video from the system to open it up and start analysis\n")
	print("\n2. You can play/pause video, step a frame forward/backwards, segment any object using SAM with text prompt and placing markers on the image\n")

	return 

def main():
	print_how_to()

	va.torch.backends.cuda.matmul.allow_tf32 = True
	va.torch.backends.cudnn.allow_tf32 = True
	va.torch.autocast("cuda", dtype=va.torch.bfloat16).__enter__()

	device = 'cuda' if va.torch.cuda.is_available() else 'cpu'
	model = va.Sam3Model.from_pretrained(va.SAM_MODEL).to(device)
	processor = va.Sam3Processor.from_pretrained(va.SAM_MODEL)

	while True:
		app = va.VideoAnalyzer(model, processor, device)
		if not hasattr(app, 'cap') or app.cap is None:
			break
			
		open_another = app.run()
		if not open_another:
			break    

if __name__ == "__main__":    
	main()