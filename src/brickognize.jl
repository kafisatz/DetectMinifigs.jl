using DetectMinifigs
using Images; using ONNXRunTime;using Genie;using Genie.Router;using Genie.Requests;using Genie.Renderer.Json
using JSON3 
#good example, score is 90pct
fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\2024090614.jpg"

fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\20240906119.jpg"
fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\20240906112.jpg"

js = brickognize(fi)
nitems = size(js.items,1)
js.items[1]



fldr = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person"
fis = readdir(fldr)
@show tmpdr = mktempdir()
count = 0
for firel in fis
    @show count+=1
    fi = joinpath(fldr, firel)
    brickognize_process_file(fi,tmpdr)
end

