using DetectMinifigs
using Images; using ONNXRunTime;using Genie;using Genie.Router;using Genie.Requests;using Genie.Renderer.Json

#run main function
DetectMinifigs.main(DetectMinifigs.mdl_path)

#=
userhome = ENV["USERPROFILE"]
fldr = joinpath(userhome,raw"OneDrive - K\Dateien\Lego\minifigs\20240906.jpg")
@assert isfile(fi) "File not found: $fi"
input, img_width, img_height = prepare_input(fi)
output = run_model(input,mdl_path)
res = process_output(output, img_width, img_height)


download_model_if_not_exists(mdl_path)
=#