"""Edit a physics world directly, without HTTP; useful for custom experiments."""

from pathlib import Path

from PIL import Image

from humaned_lab.simulation.world import PhysicsWorld

scene = Path(__file__).resolve().parents[2] / "configs/scenes/moving_cubes.json"
output = Path("outputs/custom-world.png")
output.parent.mkdir(exist_ok=True)
with PhysicsWorld(scene) as world:
    world.set_cube("target", [0.25, 0.15, 0.30], [0.10, 0, 0.25])
    for _ in range(1000):
        world.step()
    print(world.state())
    Image.fromarray(world.render()).save(output)
