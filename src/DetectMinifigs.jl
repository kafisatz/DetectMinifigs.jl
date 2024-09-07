module DetectMinifigs

using Images; using ONNXRunTime;using Genie;using Genie.Router;using Genie.Requests;using Genie.Renderer.Json; using JSON3
using libwebp_jll
using Downloads

export mdl_path
mdl_path = normpath(joinpath(@__DIR__,"..","model","yolov8x.onnx"))
#isfile(mdl_path)

export brickognize 
function brickognize(fi)
    @assert isfile(fi)
    #s flag is for silent -> suppress progress bar
    cmd = `curl -s -X 'POST' https://api.brickognize.com/predict/figs/ -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F "query_image=@$fi;type=image/jpeg"`
    rs = read(pipeline(cmd));
    js = JSON3.read(rs)


    #run with pipeline
    #rs = run(pipeline(cmd,stdout=Base.DevNull()));
    #out = IOBuffer()
    #rs = run(pipeline(cmd,stdout=out));
    #out = IOBuffer()
    #rs = run(pipeline(cmd,stdout=out));
    #read(out,String)
    #rs = read(cmd,String,stdout=Base.DevNull())) #prints to console
    
return js
end


# Main function, starts the web server
export main 
function main(mdl_path)    

    download_model_if_not_exists(mdl_path)
    # handle site root. Returns index.html file content
    route("/") do 
        String(read("index.html"))
    end 

    #yolo parameters
    #iou=0.1 conf=0.01 classes=0 save_crop=True source="C:\Users\bernhard.koenig\OneDrive - K\Dateien\Lego\minifigs\20240906.jpg"
    yolo_settings = (iou=0.1, conf=0.01, classes=0, save_crop=true)

    # handle /detect POST requests: receive an image from frontend
    # and returns array of detected objects in a format: [x1,y1,x2,y2,class,probability]
    # to a web browser as a JSON
    route("/detect", method=POST) do
        buf = IOBuffer(filespayload()["image_file"].data)
        json(detect_objects_on_image(buf,mdl_path))
    end

    up(8080, host="0.0.0.0", async=false)
end

# Function receives an uploaded image file body
# passes it through the YOLOv8 neural network
# model and returns an array of bounding boxes of 
# detected objects where each bounding box is an 
# array in a format [x1,y1,x2,y2,object_class,probability]
export detect_objects_on_image 
function detect_objects_on_image(buf,mdl_path)
    input, img_width, img_height = prepare_input(buf)
    output = run_model(input,mdl_path)
    return process_output(output, img_width, img_height)
end

# Function resizes image to a size,
# supported by default Yolov8 neural network (640x640)
# converts in to a tensor of (1,3,640,640) shape
# that supported as an input to a neural network
export prepare_input 
function prepare_input(buf)
    img = load(buf)
    img_height, img_width = size(img)
    img = imresize(img,(640,640))
    img = RGB.(img)
    input = channelview(img)
    input = reshape(input,1,3,640,640)
    return Float32.(input), img_width, img_height    
end

# Function receives an input image tensor,
# passes it through the YOLOv8 neural network
# model and returns the raw object detection
# result
export run_model 
function run_model(input,mdl_path)
    @assert isfile(mdl_path) "Model file not found: $mdl_path"
    model = load_inference(mdl_path)
    outputs = model(Dict("images" => input))
    #=
        outputs = model(Dict("images" => input),confthres=0.1)
        @edit model(Dict("images" => input))

        @edit load_inference(mdl_path)
        load_inference(mdl_path,confidence_thres=0.1)
    =#
    return outputs["output0"]
end

# Function receives the raw object detection
# result from neural network and converts
# it to an array of bounding boxes. Each
# bounding box is an array of the following format:
# [x1,y1,x2,y2,object_class,probability]
export process_output
function process_output(output, img_width, img_height)
    output = output[1,:,:]
    output = transpose(output)

    boxes = []
    for row in eachrow(output)
        prob = maximum(row[5:end])
        if prob<0.5
            continue
        end
        class_id = Int(argmax(row[5:end]))
        label = yolo_classes[class_id]
        xc,yc,w,h = row[1:4]
        x1 = (xc-w/2)/640*img_width
        y1 = (yc-h/2)/640*img_height
        x2 = (xc+w/2)/640*img_width
        y2 = (yc+h/2)/640*img_height
        push!(boxes,[x1,y1,x2,y2,label,prob])
    end

    boxes = sort(boxes, by = item -> item[6], rev=true)
    result = []
    while length(boxes)>0
        push!(result,boxes[1])
        boxes = filter(box -> iou(box,boxes[1])<0.7,boxes)
    end
    return result
end

# Calculates "Intersection-over-union" coefficient
# for specified two boxes
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
# Each box should be provided in a format:
# [x1,y1,x2,y2,object_class,probability]
function iou(box1,box2)
    return intersection(box1,box2) / union(box1,box2)
end

# Calculates union area of two boxes
# Each box should be provided in a format:
# [x1,y1,x2,y2,object_class,probability]
function union(box1,box2)
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[1:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[1:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)
end

# Calculates intersection area of two boxes
# Each box should be provided in a format:
# [x1,y1,x2,y2,object_class,probability]
function intersection(box1,box2)
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[1:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[1:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)
end

# Array of YOLOv8 class labels
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

export download_model_if_not_exists
function download_model_if_not_exists(mdl_path)
    if isfile(mdl_path)
        return nothing 
    end 

    @info("Downloading model file...")
    #DetectMinifigs
    pkgpath = normpath(joinpath(splitdir(pathof(DetectMinifigs))[1],".."))
    cmd = `python src/download_and_convert_model_to_onnx.py`
    run(cmd)
    return nothing
end

#main(mdl_path)

export download_webp_to_image
function download_webp_to_image(url)
    webp = Downloads.download(url)
    png = splitext(webp)[1] * ".png"
    libwebp_jll.dwebp() do dwebp
        run(`$dwebp -quiet $webp -o $png`)
    end
    img = load(png)
    rm(webp)
    rm(png)

    #save(raw"C:\temp\abc.png",img)
    #download_webp_to_image(url)
    #save(raw"C:\temp\abc.png",img)
    return img
end


export concatenate_images
function concatenate_images(img,matchedimg)
    h1,w1 = size(img)
    h2,w2 = size(matchedimg)

    #align width
    #make smaller image wider
    if w1 == w2 
        return vcat(img,matchedimg)
    end
    
    new_width = max(w1,w2)
    if new_width == w2
        percentage_scale = w2 / w1
        new_size = trunc.(Int, size(img) .* percentage_scale)
        new_size = (new_size[1], w2) #second dim must be w2!
        img_rescaled = imresize(img, new_size)
        return vcat(img_rescaled,matchedimg)
    else
        percentage_scale = w1 / w2
        new_size = trunc.(Int, size(matchedimg) .* percentage_scale)
        new_size = (new_size[1], w1) #second dim must be w1!
        img_rescaled = imresize(matchedimg, new_size)
        return vcat(img,img_rescaled)
    end

return nothing 
end

export brickognize_process_file
function brickognize_process_file(fi,outputdir)
    @assert isdir(outputdir)
    #println(fi)
    fnonly = splitdir(fi)[2]
    js = brickognize(fi)
    nitems = size(js.items,1)
    item = js.items[1]
    img_url = item.img_url
    score = item.score
    println(score)

    @assert endswith(img_url,".webp")
    matchedimg = download_webp_to_image(img_url)
    #save(raw"C:\temp\abc.png",img)

    img = Images.load(fi)
    img_concatenated = concatenate_images(img, matchedimg)
    pt = joinpath(outputdir,"score_$(round(score*100))_$(fnonly).png")
    save(pt,img_concatenated)    
    return nothing 
end


end #end Module
