using Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

using JSON3
using PackageCompiler
using SHA
using StructTypes

const PRECOMPILE_WORKLOAD = joinpath(@__DIR__, "precompile_workload.jl")

function default_sysimage_path()
    extension = Sys.isapple() ? ".dylib" : (Sys.iswindows() ? ".dll" : ".so")
    return joinpath(@__DIR__, "forecastflows_sysimage" * extension)
end

function write_metadata(sysimage_path::String)
    manifest_path = joinpath(@__DIR__, "Manifest.toml")
    metadata_path = sysimage_path * ".json"
    manifest_sha256 = bytes2hex(sha256(read(manifest_path)))
    metadata = Dict(
        "julia_major_minor" => "$(VERSION.major).$(VERSION.minor)",
        "manifest_sha256" => manifest_sha256,
        "built_at_unix_secs" => floor(Int, time()),
    )
    open(metadata_path, "w") do io
        JSON3.write(io, metadata)
    end
    return metadata_path
end

sysimage_path = get(ENV, "FORECASTFLOWS_SYSIMAGE_OUTPUT", default_sysimage_path())
mkpath(dirname(sysimage_path))

create_sysimage(
    [:ForecastFlows, :JSON3, :StructTypes];
    project=@__DIR__,
    sysimage_path=sysimage_path,
    precompile_execution_file=PRECOMPILE_WORKLOAD,
)

metadata_path = write_metadata(sysimage_path)
println("built ForecastFlows sysimage: " * sysimage_path)
println("wrote sysimage metadata: " * metadata_path)
