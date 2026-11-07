# Paper-facing package documentation coverage.
@testset "exported API documentation coverage" begin
    documented_modules = (Ptd, Ptd.DcEgm, Ptd.G2EGM, Ptd.IID, Ptd.AR1, Ptd.HSZ)
    missing = String[]
    for mod in documented_modules
        for name in names(mod; all=false, imported=false)
            isdefined(mod, name) || continue
            value = getfield(mod, name)
            value isa Function || value isa UnionAll || value isa DataType ||
                continue
            Base.Docs.hasdoc(mod, name) || push!(missing, "$(mod).$(name)")
        end
    end
    @test isempty(missing)
end
