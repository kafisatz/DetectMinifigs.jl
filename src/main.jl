using DetectMinifigs
using Images; using ONNXRunTime;using Genie;using Genie.Router;using Genie.Requests;using Genie.Renderer.Json

#run main function
DetectMinifigs.main(DetectMinifigs.mdl_path)

#=
using Pkg
Pkg.add("Images")
Pkg.add("ONNXRunTime")
Pkg.add("Genie")
=#


#=
fi = raw"C:\Users\BernhardKönig\OneDrive - K\Dateien\Lego\minifigs\20240906.jpg"
@assert isfile(fi) "File not found: $fi"
p = prepare_input(fi)
=#