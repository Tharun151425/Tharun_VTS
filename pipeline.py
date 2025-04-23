# import cv2
# from tkinter import *
# from tkinter import messagebox, filedialog
# from inference import InferencePipeline

# class Application:
#     def __init__(self, window):
#         self.window = window
#         self.window.title("Roboflow Inference GUI")
        
#         # Start video capture
#         self.capture = cv2.VideoCapture(0)
#         self.api_key = "YOUR_API_KEY"
#         self.workspace_name = "YOUR_WORKSPACE_NAME"
#         self.workflow_id = "YOUR_WORKFLOW_ID"

#         # Initialize InferencePipeline
#         self.pipeline = InferencePipeline.init_with_workflow(
#             api_key=self.api_key,
#             workspace_name=self.workspace_name,
#             workflow_id=self.workflow_id,
#             video_reference=0,
#             on_prediction=self.process_result
#         )

#         # Create a Tkinter button to start the application
#         start_button = Button(window, text="Start Inference", command=self.start_inference)
#         start_button.pack()

#         # Create a Tkinter button to stop the application
#         stop_button = Button(window, text="Stop Inference", command=self.stop_inference)
#         stop_button.pack()

#     def start_inference(self):
#         self.pipeline.start()

#     def stop_inference(self):
#         self.pipeline.stop()
#         self.capture.release()
#         cv2.destroyAllWindows()

#     def process_result(self, result, video_frame):
#         # Display video frame with the detection results
#         cv2.imshow("Roboflow Inference", video_frame)

# # Run the application
# if __name__ == "__main__":
#     root = Tk()
#     app = Application(root)
#     root.mainloop()

# Import the InferencePipeline object
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result) # do something with the predictions of each frame


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="rYiJwrogVwGCyVKQ7VzN",
    workspace_name="mrfury",
    workflow_id="detect-and-classify",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish