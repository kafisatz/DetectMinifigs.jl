using DetectMinifigs
using Images; using ONNXRunTime;using Genie;using Genie.Router;using Genie.Requests;using Genie.Renderer.Json;using JSON3;import GMT
#good example, score is 90pct
fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\2024090614.jpg"

fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\2024090635.jpg"

#bad examples
fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\20240906119.jpg"
fi = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person\20240906112.jpg"

js = brickognize(fi)
nitems = size(js.items,1)
js.items[1]

#score of 50 seems OK, generally
fldr = raw"C:\Users\bernhard.koenig\runs\detect\predict14\crops\person"
fis = readdir(fldr)
@show outputdir = mktempdir(raw"C:\temp")
count = 0
sz = size(fis,1)
for firel in fis
    count+=1
    fi = joinpath(fldr, firel)
    @show fi
    @show count/sz
    id,name,score,js = brickognize_process_file(fi,outputdir);
end