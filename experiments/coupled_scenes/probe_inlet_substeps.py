"""Short inlet diagnostic using the production emitter unchanged.

Run with Isaac Sim python.bat. This writes diagnostic data, not a valid
production cache: no bodies are present; optional guide walls test inlet contacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def regular_layer_births(source, frames):
    """Diagnostic whole-layer schedule, retaining spacing and average flux."""
    import numpy as np

    spacing = float(source["spacing"])
    velocity = np.asarray(source["velocity"], dtype=np.float64)
    if velocity[0] != 0 or velocity[2] != 0 or velocity[1] >= 0:
        raise ValueError("Regular-layer control requires a downward vertical inlet")
    period_steps = spacing / abs(velocity[1]) * 720
    step_stride = int(round(period_steps))
    if step_stride < 1 or not np.isclose(period_steps, step_stride):
        raise ValueError("Layer period must align exactly with diagnostic substeps")
    axes = []
    for axis in (0, 2):
        count = max(1, int(np.floor(source["size"][axis] / spacing)))
        axes.append(source["centre"][axis] + (np.arange(count) - .5 * (count - 1)) * spacing)
    xz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    positions = np.empty((len(xz), 3), dtype=np.float32)
    positions[:, (0, 2)] = xz
    positions[:, 1] = source["centre"][1]
    velocities = np.tile(velocity.astype(np.float32), (len(xz), 1))
    return [dict(birth_frame=step // 12 + 1, birth_substep=step % 12,
                 positions=positions.copy(), velocities=velocities.copy(),
                 ids=np.arange(i * len(xz), (i + 1) * len(xz), dtype=np.int64))
            for i, step in enumerate(range(0, frames * 12, step_stride))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--particle-contact-offset", type=float)
    parser.add_argument("--neighborhood-scale", type=float)
    parser.add_argument("--cfl-coefficient", type=float)
    parser.add_argument("--block-layers", type=int, default=12)
    parser.add_argument("--initial-state", type=Path, help="Import snapshot positions only, recentered without scaling; retain uniform configured birth velocity.")
    parser.add_argument("--repeat-block-every-frames", type=int, help="Diagnostic periodic full-block births through the requested duration.")
    parser.add_argument("--repeat-block-every-substeps", type=int, help="Repeat a single-chunk template on an explicit substep cadence.")
    parser.add_argument("--downward-speed", type=float)
    parser.add_argument("--upstream-height", type=float, default=0.0, help="Raise the birth plane only; no walls or velocity resets.")
    parser.add_argument("--guide-walls", action="store_true", help="Four static 10 mm walls, 88x64 mm bore, 40 mm headroom; no end caps.")
    parser.add_argument("--kill-above-guide", action="store_true", help="Remove particles above the guide top after each solve; no replacement births.")
    parser.add_argument("--birth-spacing", type=float, help="Volume-block position spacing only; leaves all native offsets and mass unchanged.")
    parser.add_argument("--birth-layout", choices=("original", "regular_layers", "volume_block"), default="original")
    parser.add_argument("--material-control", choices=("unchanged", "no_cohesion", "no_surface_tension", "neither", "no_viscosity", "no_vorticity", "density_only"), default="unchanged")
    args = parser.parse_args()
    if args.kill_above_guide and not args.guide_walls:
        raise ValueError("Top removal requires guide walls")
    if args.guide_walls and (args.upstream_height <= 0 or args.birth_layout != "original"):
        raise ValueError("Guide walls require a raised original inlet")
    if args.repeat_block_every_substeps is not None and (args.repeat_block_every_substeps < 1 or args.birth_layout != "volume_block" or args.repeat_block_every_frames is not None):
        raise ValueError("Substep repetition requires volume_block and no frame repetition")
    if args.repeat_block_every_frames is not None and (args.repeat_block_every_frames < 1 or args.birth_layout != "volume_block"):
        raise ValueError("Periodic blocks require volume_block and a positive interval")
    if args.block_layers < 1 or args.block_layers > 48:
        raise ValueError("Block thickness must be 1..48 layers")
    if args.block_layers != 12 and args.birth_layout != "volume_block":
        raise ValueError("Block thickness requires volume_block")
    import math
    if not math.isfinite(args.upstream_height) or not 0 <= args.upstream_height <= .2:
        raise ValueError("Upstream height must be between 0 and 0.2 m")
    for value in (args.particle_contact_offset, args.neighborhood_scale, args.cfl_coefficient, args.downward_speed):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Diagnostic overrides must be finite and positive")
    if args.neighborhood_scale is not None and args.neighborhood_scale <= 1:
        raise ValueError("Neighborhood scale must exceed 1")
    if args.frames < 1 or args.frames > 15:
        raise ValueError("Diagnostic is limited to 1..15 output frames")
    if args.initial_state is not None and (args.birth_layout != "volume_block" or args.birth_spacing is not None):
        raise ValueError("Snapshot import requires volume_block and no spacing override")
    if args.birth_spacing is not None:
        import math
        if args.birth_layout != "volume_block" or not math.isfinite(args.birth_spacing) or args.birth_spacing <= 0:
            raise ValueError("--birth-spacing must be positive and requires volume_block")
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 64, "height": 64})
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils, Sdf, Vt
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from coupled_scene.pbd_pour_event import PbdPourEvent

    config = json.loads(args.configuration.read_text())
    diagnostic_outlet_y = float(config["source"]["centre"][1])
    config["source"]["centre"][1] += args.upstream_height
    if args.downward_speed is not None:
        config["source"]["velocity"] = [0.0, -args.downward_speed, 0.0]
    config["source"].update(start_frame=1, stop_frame=args.frames)
    if args.birth_layout == "volume_block":
        # One 21 x 12 x 15 lattice for the v74 nozzle: 3780 particles,
        # 4 mm apart in all axes. All start together with identical velocity.
        # No following births or ballistic layer back-timing are involved.
        config["source"]["emission_model"] = "static_slab"
        config["source"]["stop_frame"] = 1
        if args.repeat_block_every_frames is not None:
            config["source"]["stop_frame"] = args.frames
            config["source"]["interval_frames"] = args.repeat_block_every_frames
        config["source"]["size"][1] = args.block_layers * config["source"]["spacing"]
    if args.solver_iterations is not None:
        if args.solver_iterations < 1:
            raise ValueError("solver iterations must be positive")
        config["source"]["solver_position_iterations"] = args.solver_iterations
    if args.material_control in {"no_cohesion", "neither"}:
        config["source"]["cohesion"] = 0.0
    if args.material_control in {"no_surface_tension", "neither"}:
        config["source"]["surface_tension"] = 0.0
    if args.material_control == "no_viscosity":
        config["source"]["viscosity"] = 0.0
    if args.material_control == "no_vorticity":
        config["source"]["vorticity_confinement"] = 0.0
    if args.material_control == "density_only":
        for key in ("viscosity", "vorticity_confinement", "cohesion", "surface_tension", "damping", "adhesion", "friction"):
            config["source"][key] = 0.0
    config["settle_steps"] = 0
    config["production_target"] = None
    config["diagnostic_birth_layout"] = args.birth_layout
    config["diagnostic_birth_spacing_m"] = args.birth_spacing
    config["diagnostic_particle_contact_offset_m"] = args.particle_contact_offset
    config["diagnostic_neighborhood_scale"] = args.neighborhood_scale
    config["diagnostic_cfl_coefficient"] = args.cfl_coefficient
    config["diagnostic_block_layers"] = args.block_layers
    config["diagnostic_initial_state"] = str(args.initial_state) if args.initial_state else None
    config["diagnostic_repeat_block_every_frames"] = args.repeat_block_every_frames
    config["diagnostic_repeat_block_every_substeps"] = args.repeat_block_every_substeps
    config["diagnostic_upstream_height_m"] = args.upstream_height
    config["diagnostic_outlet_y_m"] = diagnostic_outlet_y
    config["diagnostic_guide_walls"] = args.guide_walls
    config["diagnostic_kill_above_guide"] = args.kill_above_guide
    config_path = args.output / "probe_configuration.json"
    config_path.write_text(json.dumps(config, indent=2))
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, -1, 0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    api.CreateEnableGPUDynamicsAttr().Set(True)
    api.CreateBroadphaseTypeAttr().Set("GPU")
    api.CreateSolverTypeAttr().Set("TGS")
    api.CreateTimeStepsPerSecondAttr().Set(720)
    api.CreateMaxBiasCoefficientAttr().Set(240.0)
    api.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    event = PbdPourEvent(stage, scene, config_path, args.output, args.frames, 12)
    material = PhysxSchema.PhysxPBDMaterialAPI(stage.GetPrimAtPath("/World/CoupledEvent/Looks/WaterPhysics"))
    if args.cfl_coefficient is not None:
        material.CreateCflCoefficientAttr().Set(args.cfl_coefficient)
    diagnostic_cfl = material.GetCflCoefficientAttr().Get()
    assert diagnostic_cfl is not None
    particle_system = PhysxSchema.PhysxParticleSystem.Get(stage, "/World/CoupledEvent/ParticleSystem")
    if args.particle_contact_offset is not None:
        assert args.particle_contact_offset > max(particle_system.GetSolidRestOffsetAttr().Get(), particle_system.GetFluidRestOffsetAttr().Get())
        particle_system.GetParticleContactOffsetAttr().Set(args.particle_contact_offset)
    if args.neighborhood_scale is not None:
        particle_system.GetNeighborhoodScaleAttr().Set(args.neighborhood_scale)
    diagnostic_system = {name: particle_system.GetPrim().GetAttribute(name).Get()
                         for name in ("particleContactOffset", "solidRestOffset", "fluidRestOffset", "restOffset", "contactOffset", "neighborhoodScale", "maxNeighborhood", "solverPositionIterationCount")}
    if args.initial_state is not None:
        with np.load(args.initial_state) as snapshot:
            positions = snapshot["positions"].astype(np.float64)
            snapshot_ids = snapshot["particle_ids"].copy()
        for record in event.batch_records:
            assert np.array_equal(snapshot_ids, record["ids"] - record["ids"][0])
            assert positions.shape == record["positions"].shape and np.isfinite(positions).all()
            translation = record["positions"].astype(np.float64).mean(axis=0) - positions.mean(axis=0)
            record["positions"] = (positions + translation).astype(np.float32)
        (args.output / "initial_state_import.json").write_text(json.dumps(dict(
            source=str(args.initial_state), translation_m=translation.tolist(),
            count=len(positions), velocity_policy="configured_uniform_birth_velocity",
            geometry_policy="translation_only_no_scaling"), indent=2))
    if args.birth_spacing is not None:
        if event.particle_set_strategy != "chunked_density_sets":
            raise ValueError("Position-spacing control requires chunked_density_sets")
        # Native offsets and mass were already configured at original spacing.
        # Preserve particle count/IDs/velocities; expand only birth positions.
        centre = np.asarray(config["source"]["centre"], dtype=np.float64)
        scale = args.birth_spacing / config["source"]["spacing"]
        for record in event.batch_records:
            record["positions"] = (centre + (record["positions"].astype(np.float64) - centre) * scale).astype(np.float32)
    if args.birth_layout == "regular_layers":
        # Reuse the exact same native density-set append path. This diagnostic
        # changes only birth positions/times, not production emitter code.
        if event.particle_set_strategy != "chunked_density_sets":
            raise ValueError("Regular-layer control requires chunked_density_sets")
        records = regular_layer_births(config["source"], args.frames)
        count = sum(len(record["ids"]) for record in records)
        if count > config.get("emission_chunk_particles", 16384):
            raise ValueError("Short regular-layer control must fit in one chunk")
        assert count == sum(len(record["ids"]) for record in event.batch_records)
        template = event.batch_records[0]
        event.batch_records = [dict(template, **record, starts_chunk=i == 0)
                               for i, record in enumerate(records)]
    if args.repeat_block_every_substeps is not None:
        assert event.particle_set_strategy == "chunked_density_sets"
        assert len(event.batch_records) == 1
        template = event.batch_records[0]
        count = len(template["ids"])
        starts = list(range(0, args.frames * 12, args.repeat_block_every_substeps))
        assert count * len(starts) <= config.get("emission_chunk_particles", 8192)
        event.batch_records = [dict(template, birth_frame=step // 12 + 1, birth_substep=step % 12,
                                   positions=template["positions"].copy(), velocities=template["velocities"].copy(),
                                   ids=np.arange(i * count, (i + 1) * count, dtype=np.int64), starts_chunk=i == 0)
                               for i, step in enumerate(starts)]
        template["instancer"].GetPrim().GetAttribute("physxParticle:maxParticles").Set(count * len(starts))
    # Remove production cabinet; optionally add just the diagnostic guide walls.
    stage.RemovePrim("/World/CoupledEvent/GlassCabinet")
    guide = None
    if args.guide_walls:
        from pxr import UsdShade
        from omni.physx.scripts import physicsUtils
        width, depth, thickness, headroom = .088, .064, .010, .040
        lower, upper = diagnostic_outlet_y, config["source"]["centre"][1] + headroom
        cx, _, cz = config["source"]["centre"]
        cy, height = (lower + upper) / 2, upper - lower
        mat = UsdShade.Material.Define(stage, "/World/GuideWallMaterial")
        mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mat_api.CreateStaticFrictionAttr().Set(.05)
        mat_api.CreateDynamicFrictionAttr().Set(.05)
        mat_api.CreateRestitutionAttr().Set(0.0)
        panels = [
            ("Left", (cx-(width+thickness)/2,cy,cz), (thickness,height,depth+2*thickness)),
            ("Right", (cx+(width+thickness)/2,cy,cz), (thickness,height,depth+2*thickness)),
            ("Front", (cx,cy,cz-(depth+thickness)/2), (width,height,thickness)),
            ("Back", (cx,cy,cz+(depth+thickness)/2), (width,height,thickness)),
        ]
        for name, centre, size in panels:
            cube = UsdGeom.Cube.Define(stage, "/World/Guide/" + name)
            cube.CreateSizeAttr().Set(1.)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateRestOffsetAttr().Set(0.)
            collision.CreateContactOffsetAttr().Set(.001)
            physicsUtils.add_physics_material_to_prim(stage, cube.GetPrim(), mat.GetPath())
        guide = dict(inner_width_m=width, inner_depth_m=depth, wall_thickness_m=thickness,
                     lower_y_m=lower, upper_y_m=upper, headroom_m=headroom,
                     wall_rest_offset_m=0., wall_contact_offset_m=.001, restitution=0.)
    collider_count = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())
    assert collider_count == (4 if args.guide_walls else 0)
    app.update()
    stage.GetRootLayer().Export(str(args.output / "probe_stage.usda"))
    simulation = get_physx_simulation_interface()
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)
    simulation.attach_stage(stage_id.ToLongInt())
    rows = []
    removed_ids_all = set()
    dt = 1.0 / 720.0

    def summary(positions, velocities):
        speed = np.linalg.norm(velocities, axis=1)
        return dict(count=len(positions), max_vy=float(velocities[:, 1].max()),
                    min_vy=float(velocities[:, 1].min()),
                    max_speed=float(speed.max()), rms_speed=float(np.sqrt(np.mean(speed**2))),
                    upward_count=int((velocities[:, 1] > 0).sum()),
                    up_over_2_count=int((velocities[:, 1] > 2).sum()))

    try:
        for step in range(args.frames * 12):
            frame, substep = step // 12 + 1, step % 12
            before_p, before_v, before_ids = (a.copy() for a in event.state_arrays())
            record = (event.batch_records[event.enabled_batches]
                      if event.enabled_batches < len(event.batch_records) else None)
            if record is not None and (record["birth_frame"], record["birth_substep"]) == (frame, substep):
                new_p, new_v = record["positions"], record["velocities"]
            else:
                new_p = np.empty((0, 3), dtype=np.float32)
                new_v = np.empty((0, 3), dtype=np.float32)
            row = dict(step=step, frame=frame, substep=substep, new_count=len(new_p))
            if len(before_p) and len(new_p):
                distances, indices = cKDTree(before_p).query(new_p)
                row.update(min_new_old_distance_m=float(distances.min()),
                           new_with_old_within_2mm=int((distances < .002).sum()),
                           nearest_old_id=int(before_ids[indices[np.argmin(distances)]]))
            event.enable_due(frame, substep, app)
            inserted_p, inserted_v, ids = (a.copy() for a in event.state_arrays())
            row["after_enable"] = summary(inserted_p, inserted_v)
            n = len(before_p)
            row["enable_old_position_change_max_m"] = float(np.max(np.linalg.norm(inserted_p[:n]-before_p, axis=1), initial=0))
            row["enable_old_velocity_change_max"] = float(np.max(np.linalg.norm(inserted_v[:n]-before_v, axis=1), initial=0))
            simulation.simulate(dt, step * dt)
            simulation.fetch_results()
            fetched_p, fetched_v, _ = (a.copy() for a in event.state_arrays())
            app.update()
            after_p, after_v, after_ids = (a.copy() for a in event.state_arrays())
            row["after_solve"] = summary(after_p, after_v)
            row["post_fetch_update_position_change_max_m"] = float(np.max(np.linalg.norm(after_p-fetched_p, axis=1), initial=0))
            row["post_fetch_update_velocity_change_max"] = float(np.max(np.linalg.norm(after_v-fetched_v, axis=1), initial=0))
            for array in (after_p, after_v):
                assert np.isfinite(array).all(), row
            removed_ids = np.empty(0, dtype=np.int64)
            if args.kill_above_guide:
                assert event.particle_set_strategy == "chunked_density_sets"
                kill = after_p[:, 1] > guide["upper_y_m"]
                removed_ids = after_ids[kill]
                if len(removed_ids):
                    assert not removed_ids_all.intersection(removed_ids.tolist())
                    cursor = 0
                    with Sdf.ChangeBlock():
                        for instancer in event.active_instancers:
                            prim = instancer.GetPrim()
                            mass_api = UsdPhysics.MassAPI(prim)
                            assert not mass_api.GetMassAttr().HasAuthoredValueOpinion()
                            assert mass_api.GetDensityAttr().Get() == config["source"]["density"]
                            count = len(instancer.GetPositionsAttr().Get())
                            keep = ~kill[cursor:cursor+count]
                            if not keep.all():
                                instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(after_p[cursor:cursor+count][keep]))
                                instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(after_v[cursor:cursor+count][keep]))
                                proto = np.asarray(instancer.GetProtoIndicesAttr().Get(),dtype=np.int32)
                                instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(proto[keep]))
                            cursor += count
                    assert cursor == len(after_ids)
                    event.active_id_chunks = [chunk[~np.isin(chunk,removed_ids)] for chunk in event.active_id_chunks]
                    app.update()
                    kept_p, kept_v, kept_ids = event.state_arrays()
                    assert np.array_equal(kept_ids,after_ids[~kill])
                    assert np.array_equal(kept_p,after_p[~kill])
                    assert np.array_equal(kept_v,after_v[~kill])
                    removed_ids_all.update(removed_ids.tolist())
                row["removed_count"] = len(removed_ids)
                row["retained_count"] = len(after_ids) - len(removed_ids)
                row["removed_total"] = len(removed_ids_all)
            np.savez_compressed(args.output / f"substep_{step:03d}.npz",
                                before_positions=before_p, before_velocities=before_v,
                                new_positions=new_p, new_velocities=new_v,
                                inserted_positions=inserted_p, inserted_velocities=inserted_v,
                                positions=after_p, velocities=after_v, particle_ids=after_ids,
                                removed_particle_ids=removed_ids)
            rows.append(row)
            print(json.dumps(row), flush=True)
        report = dict(complete=True, collider_count=collider_count, guide=guide, dt=dt, frames=args.frames,
                      kill_above_guide=args.kill_above_guide, removed_total=len(removed_ids_all),
                      material_control=args.material_control,
                      birth_layout=args.birth_layout,
                      birth_spacing_m=args.birth_spacing or config["source"]["spacing"],
                      solver_iterations=config["source"]["solver_position_iterations"],
                      particle_system_attributes=diagnostic_system,
                      cfl_coefficient=diagnostic_cfl,
                      block_layers=args.block_layers,
                      initial_state=str(args.initial_state) if args.initial_state else None,
                      repeat_block_every_frames=args.repeat_block_every_frames,
                      repeat_block_every_substeps=args.repeat_block_every_substeps,
                      source_configuration=str(args.configuration), steps=rows)
        (args.output / "inlet_probe_report.json").write_text(json.dumps(report, indent=2))
    finally:
        simulation.detach_stage()
        app.close()


if __name__ == "__main__":
    main()
