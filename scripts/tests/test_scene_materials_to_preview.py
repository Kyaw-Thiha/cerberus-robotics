"""Unit tests for the MDL -> UsdPreviewSurface converter, on a synthetic stage (no scene data, no GPU)."""

from pathlib import Path

import pytest
import scene_materials_to_preview as conv

pxr = pytest.importorskip("pxr", reason="pxr (usd-core / Isaac Sim python) is needed to author USD")
from pxr import Sdf, Usd, UsdGeom, UsdShade


def _make_scene(root: Path, *, missing_texture: bool = False) -> Path:
    tex_dir = root / "Textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    diffuse = tex_dir / "wall_diffuse.png"
    if not missing_texture:
        diffuse.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")
    normal = tex_dir / "wall_normal.png"
    normal.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")

    path = root / "start_result_navigation.usd"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")

    mat = UsdShade.Material.Define(stage, "/World/Materials/M_Wall")
    shader = UsdShade.Shader.Define(stage, "/World/Materials/M_Wall/mdl_surface")
    shader.CreateIdAttr("mdl:surface")
    shader.CreateInput("BaseColorTexture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(diffuse) if not missing_texture else "Textures/absent.png"))
    shader.CreateInput("NormalTexture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(normal)))
    shader.CreateInput("BaseColor", Sdf.ValueTypeNames.Float3).Set((0.2, 0.3, 0.4))
    shader.CreateInput("Roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("Materials/Wall.mdl"))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")

    mesh = UsdGeom.Mesh.Define(stage, "/World/Geometry/Wall")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0, 0), (1, 0), (1, 1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    return path


def test_classify_inputs():
    assert conv.classify_asset_input("BaseColorTexture") == "diffuse"
    assert conv.classify_asset_input("NormalTexture") == "normal"
    assert conv.classify_asset_input("RoughnessMap") == "roughness"
    assert conv.classify_asset_input("MetallicTexture") == "metallic"
    assert conv.classify_color_input("BaseColor") == "diffuse"
    assert conv.classify_color_input("NormalScale") is None


def test_build_layer_converts_materials_and_rebinds(tmp_path):
    scene = _make_scene(tmp_path)
    layer = tmp_path / "materials_preview.usda"
    report = conv.build_layer(scene, layer)

    assert report["materials"] == 1
    assert report["bindings"] == 1
    assert report["rebound"] == 1
    assert report["materials_with_diffuse_texture"] == 1
    assert report["materials_with_normal_texture"] == 1
    assert report["missing_texture_count"] == 0
    assert report["uv_primvar"] == "st"

    preview = Usd.Stage.Open(str(layer))
    mat_path = report["samples"][0]["preview"]
    assert preview.GetPrimAtPath(mat_path).IsA(UsdShade.Material)
    shader = preview.GetPrimAtPath(f"{mat_path}/PreviewSurface")
    assert shader.GetAttribute("info:id").Get() == "UsdPreviewSurface"
    tex = preview.GetPrimAtPath(f"{mat_path}/diffuseTex")
    assert tex.GetAttribute("inputs:file").Get().path.endswith("wall_diffuse.png")

    wall = preview.GetPrimAtPath("/World/Geometry/Wall")
    rel = wall.GetRelationship("material:binding")
    assert [str(t) for t in rel.GetTargets()] == [mat_path]


def test_original_stage_is_unchanged(tmp_path):
    scene = _make_scene(tmp_path)
    before = scene.read_bytes()
    conv.build_layer(scene, tmp_path / "materials_preview.usda")
    assert scene.read_bytes() == before  # read-only conversion: the dataset is never edited in place


def test_missing_texture_is_reported(tmp_path):
    scene = _make_scene(tmp_path, missing_texture=True)
    report = conv.build_layer(scene, tmp_path / "materials_preview.usda")
    assert report["materials_with_diffuse_texture"] == 0
    assert report["missing_texture_count"] == 1


def test_preview_wrapper_sublayers_layer_above_scene(tmp_path):
    scene = _make_scene(tmp_path)
    layer = tmp_path / "materials_preview.usda"
    conv.build_layer(scene, layer)
    wrapper = conv.write_preview_wrapper(scene, layer)

    assert wrapper.name == conv.WRAPPER_NAME
    stage = Usd.Stage.Open(str(wrapper))
    assert stage.GetPrimAtPath("/Scene").IsValid()
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    text = (tmp_path / "scene_preview.usda").read_text()
    # the override layer must compose ABOVE the scene, or the MDL materials win again
    assert text.index("materials_preview.usda") < text.index("start_result_navigation.usd")


def test_mdl_source_supplies_textures_and_colour(tmp_path):
    """GRScenes material instances name their maps inside the MDL, not in the USD shader inputs."""
    mats = tmp_path / "Materials"
    (mats / "textures" / "f2f").mkdir(parents=True)
    (mats / "textures" / "f2f" / "wall.jpg").write_bytes(b"jpeg")
    (mats / "textures" / "f2f" / "wall_n.png").write_bytes(b"png")
    (mats / "MI_wall.mdl").write_text(
        'mdl 1.7;\nimport ::KooPbr::KooMtl;\n'
        'export material MI_wall(*) = KooPbr::KooMtl(\n'
        '    diffuse: KooPbr_maps::KooPbr_bitmap(1, 0.f, 0.f, 1.f, 1.f, true, false, true, false, 0.f, 0.f, 0.f, 0, '
        'texture_2d("./textures/f2f/wall.jpg", ::tex::gamma_srgb, ""), 0.f, 1.f),\n'
        '    texmap_bump: ::unpack_normal_map(tex::lookup_float4(texture_2d("./textures/f2f/wall_n.png", '
        '::tex::gamma_linear), state::texture_coordinate(0))),\n'
        '    reflect: color(0.4f, 0.3f, 0.25f));\n'
    )
    scene = _make_scene(tmp_path)
    # re-point the synthetic material's MDL source at the instance above (the scene file lives beside Materials/)
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(scene))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Materials/M_Wall/mdl_surface"))
    for name in ("BaseColorTexture", "NormalTexture", "BaseColor"):
        shader.GetPrim().RemoveProperty(f"inputs:{name}")
    shader.GetPrim().GetAttribute("info:mdl:sourceAsset").Set(
        __import__("pxr").Sdf.AssetPath("Materials/MI_wall.mdl"))
    stage.GetRootLayer().Save()

    report = conv.build_layer(scene, tmp_path / "materials_preview.usda")
    assert report["materials_with_diffuse_texture"] == 1
    assert report["materials_with_normal_texture"] == 1
    sample = report["samples"][0]
    assert sample["textures"]["diffuse"].endswith("wall.jpg")
    assert sample["colours"]["diffuse"] == pytest.approx([0.4, 0.3, 0.25])


def test_mdl_helpers():
    text = ('diffuse: KooPbr::KooPbr_bitmap(1, 0.f, texture_2d("./textures/a/floor.jpg", ::tex::gamma_srgb, ""), 0.f)\n'
            'texmap_bump: texture_2d("./textures/a/floor_n.png", ::tex::gamma_linear)\n'
            'reflect: color(0.2f, 0.5f, 0.1f);\n')
    slots = conv.mdl_texture_slots(text)
    assert slots["diffuse"].endswith("floor.jpg")
    assert slots["normal"].endswith("floor_n.png")
    assert conv.mdl_colour(text) == (0.2, 0.5, 0.1)


def test_mdl_material_prims_are_deactivated(tmp_path):
    """Re-binding alone leaves the MDL materials in the stage and the renderer compiles every one of them."""
    scene = _make_scene(tmp_path)
    report = conv.build_layer(scene, tmp_path / "materials_preview.usda")
    assert report["mdl_materials_deactivated"] == 1
    from pxr import Usd

    stage = Usd.Stage.Open(str(tmp_path / "materials_preview.usda"))
    assert not stage.GetPrimAtPath("/World/Materials/M_Wall").IsActive()


def test_uv_set_detection_prefers_the_mesh_primvar(tmp_path):
    from pxr import Usd, UsdGeom, UsdShade

    scene = _make_scene(tmp_path)
    stage = Usd.Stage.Open(str(scene))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Geometry/Wall"))
    # GRScenes models use st1/st2 as often as st; a texture bound to "st" then samples its white fallback
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).RemovePrimvar("st")
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st1", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([(0, 0), (1, 0), (1, 1)])
    stage.GetRootLayer().Save()

    uv = conv.uv_sets_by_material(Usd.Stage.Open(str(scene)))
    assert uv["/World/Materials/M_Wall"] == "st1"


def test_instanceable_prims_are_deinstanced(tmp_path):
    """A per-instance material override is ignored while the prim is instanceable; the layer turns it off."""
    from pxr import Usd, UsdGeom

    scene = _make_scene(tmp_path)
    stage = Usd.Stage.Open(str(scene))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Geometry/Chair")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    mesh.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().Save()

    report = conv.build_layer(scene, tmp_path / "materials_preview.usda")
    assert report["instances_deinstanced"] == 1
    check = Usd.Stage.Open(str(tmp_path / "materials_preview.usda"))
    assert not check.GetPrimAtPath("/World/Geometry/Chair").IsInstanceable()


def test_preview_materials_live_inside_the_scene_root(tmp_path):
    """A target outside the referenced scope is ignored by USD, so the materials must sit under the scene root."""
    from pxr import Usd

    scene = _make_scene(tmp_path)  # default prim is /World
    report = conv.build_layer(scene, tmp_path / "materials_preview.usda")
    stage = Usd.Stage.Open(str(tmp_path / "materials_preview.usda"))
    preview_path = report["samples"][0]["preview"]
    assert preview_path.startswith("/World/__PreviewMaterials/")
    assert stage.GetPrimAtPath(preview_path).IsValid()


def test_light_scale_overrides_every_light(tmp_path):
    from pxr import Sdf, Usd, UsdLux

    scene = _make_scene(tmp_path)
    stage = Usd.Stage.Open(str(scene))
    light = UsdLux.RectLight.Define(stage, "/World/Lights/room")
    light.CreateIntensityAttr(1000.0)
    stage.GetRootLayer().Save()

    report = conv.build_layer(scene, tmp_path / "materials_preview.usda", light_scale=0.5)
    assert report["light_overrides"] == 1
    check = Usd.Stage.Open(str(tmp_path / "materials_preview.usda"))
    assert check.GetPrimAtPath("/World/Lights/room").GetAttribute("inputs:intensity").Get() == 500.0
