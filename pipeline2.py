from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"): 
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result)

pipeline = InferencePipeline.init_with_workflow(
    api_key="rYiJwrogVwGCyVKQ7VzN",
    workspace_name="mrfury",
    workflow_id="detect-and-classify",
    video_reference=0,  # 0 = default webcam
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()
