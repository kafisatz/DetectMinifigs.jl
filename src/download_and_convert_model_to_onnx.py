import os
print(os.getcwd())

pt = os.getcwd()
savepath = os.path.join(pt, 'model')
os.makedirs(savepath, exist_ok=True)
os.chdir(savepath)
print(os.getcwd())

mdl_choice = "yolov8x.pt" 
filename_onnx = mdl_choice.replace(".pt", ".onnx")
if os.path.isfile(filename_onnx):
    print("Modelfile exists. Nothing to do")
else:
    from ultralytics import YOLO
    print("creating onnx model file: ", filename_onnx)
    model = YOLO("yolov8x.pt")
    model.export(format="onnx",conf=0.01)