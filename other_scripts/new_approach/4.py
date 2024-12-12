from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Change to the desired model variant if needed



print(model.model.children())
#for x in model.model.children():
    #print("Ewfwfiwfewjfoiqjwifjwoifejweiowefwfwf")
   # print(x)

print(model.model.get_parameter())