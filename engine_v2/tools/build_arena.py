"""Build a reproducible physical sensing arena; contains no control logic."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def build(destination=ROOT/'models/reflex_arena.xml'):
    source = Path(r'C:\mujoco_models\unitree_go2')
    robot = ET.parse(source/'go2.xml').getroot()
    robot.set('model', 'Born Wired - sensory arena')
    robot.find('compiler').set('meshdir', str(source/'assets'))
    scene = ET.parse(source/'scene.xml').getroot()
    for section in scene:
        if section.tag == 'include':
            continue
        existing = robot.find(section.tag)
        if existing is not None and section.tag in ('asset', 'worldbody'):
            existing.extend(list(section))
        else:
            robot.append(section)
    world = robot.find('worldbody')
    base = world.find("body[@name='base']")
    rotation = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    # Each eye is a body with two joints, so the cameras are moved by eye
    # muscles instead of being welded to the head. At zero joint angle the
    # optical axes are exactly the fixed parallel pair this arena used before.
    # The two eyes stand 0.22 m apart. They were 0.11 m apart, and at that
    # spacing the two pictures of anything further away than about a metre were
    # the same picture - there was no disparity left for the eyes to work on -
    # which is why the whole eye loop had to be measured in a clean room with
    # one ball in it. 0.22 m is a wide pair for an animal this size, and it is
    # the widest that still leaves a target half a metre ahead inside the range
    # of offsets the binocular cells are wired for.
    quat_string = ' '.join(map(str, quat))
    for side, offset in [('left', .11), ('right', -.11)]:
        eye = ET.SubElement(base, 'body', name='eye_'+side, pos=f'.30 {offset} .05')
        ET.SubElement(eye, 'joint', name=f'eye_{side}_yaw_joint', type='hinge', axis='0 0 1',
                      range='-.6 .6', damping='.06', armature='.001', frictionloss='0')
        ET.SubElement(eye, 'joint', name=f'eye_{side}_pitch_joint', type='hinge', axis='0 1 0',
                      range='-.45 .45', damping='.06', armature='.001', frictionloss='0')
        ET.SubElement(eye, 'geom', name=f'eye_{side}_ball', type='sphere', size='.018', density='800',
                      contype='0', conaffinity='0', group='2', rgba='.12 .12 .16 1')
        ET.SubElement(eye, 'camera', name='eye_'+side, pos='0 0 0', quat=quat_string, fovy='60')
    objects = [
        ('front_block', '1.7 .20 .25', '.10 .45 .25', '.8 .35 .15 1'),
        ('left_block', '.3 1.5 .25', '.35 .12 .25', '.2 .6 .8 1'),
        ('right_block', '-.7 -1.4 .25', '.4 .12 .25', '.4 .7 .3 1'),
        ('east_wall', '3.5 0 .4', '.08 3.5 .4', '.45 .5 .55 1'),
        ('west_wall', '-3.5 0 .4', '.08 3.5 .4', '.45 .5 .55 1'),
        ('north_wall', '0 3.5 .4', '3.5 .08 .4', '.45 .5 .55 1'),
        ('south_wall', '0 -3.5 .4', '3.5 .08 .4', '.45 .5 .55 1'),
    ]
    # A flat wall carries no contrast at all, so no visual cell could ever
    # answer anything about it: not its bearings, and not whether it is
    # getting closer. It is covered by the code below so that it is something
    # to see, and so that the two eyes have the same corners to be matched on.
    for name, pos, size, rgba in objects:
        ET.SubElement(world, 'geom', name=name, type='box', pos=pos, size=size, rgba=rgba,
                      material='qrwall')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Nonperiodic visual texture supplies actual stereo correspondence.
    rng = np.random.default_rng(401)
    tex = np.repeat(np.repeat(rng.integers(70, 256, (24, 24, 1), dtype=np.uint8), 8, 0), 8, 1)
    texture_path = destination.parent/'stereo_texture.png'
    Image.fromarray(np.repeat(tex, 3, axis=2)).save(texture_path)
    asset = robot.find('asset')
    ET.SubElement(asset, 'texture', name='irregular', type='2d', file=str(texture_path))
    ET.SubElement(asset, 'material', name='textured', texture='irregular', texrepeat='1 1', texuniform='false')
    # The wall covering. It is a code: a grid of square modules, half of them
    # dark, carrying the big square marks a code wears in its corners and a few
    # smaller ones inside. Square modules are what a pair of eyes can be matched
    # on - the same corner of the same module stands in both pictures, and the
    # grain runs both ways, so a miss sideways and a miss up and down are both
    # visible. The modules are laid out from a fixed seed, so the pattern never
    # repeats, and a repeat would be indistinguishable from a match at the
    # wrong range. The two greys stay well away from black, because a wall that
    # renders as black gives the visual cells nothing to answer about.
    modules_x, modules_y, unit = 128, 16, 16
    light, dark = 235., 85.
    cells = (np.random.default_rng(20260923).random((modules_y, modules_x)) < .5).astype(np.uint8)

    def mark(top, left, size):
        """A square mark: a solid ring round a light gap round a solid middle."""
        block = np.zeros((size, size), dtype=np.uint8)
        block[0, :] = block[-1, :] = 1
        block[:, 0] = block[:, -1] = 1
        if size >= 7:
            block[2:size-2, 2:size-2] = 1
        else:
            block[size//2, size//2] = 1
        cells[top:top+size, left:left+size] = block

    for top, left in ((1, 1), (1, modules_x-8), (modules_y-8, 1)):
        mark(top, left, 7)
    for top, left in ((3, 34), (8, 62), (11, 100), (5, modules_x-30)):
        mark(top, left, 5)
    big = np.kron(cells, np.ones((unit, unit), dtype=np.uint8))
    value = np.where(big == 0, light, dark)
    shade = np.linspace(.88, 1., big.shape[0])[:, None]
    picture = np.repeat((value*shade)[..., None], 3, axis=2).astype(np.uint8)
    wall_path = destination.parent/'wall_qr.png'
    Image.fromarray(picture).save(wall_path)
    ET.SubElement(asset, 'texture', name='qrwall', type='2d', file=str(wall_path))
    ET.SubElement(asset, 'material', name='qrwall', texture='qrwall', texrepeat='1 1',
                  texuniform='false')
    props = [
        dict(name='green_target', type='sphere', pos='2 -1 .20', size='.20', rgba='.12 .85 .15 1'),
        dict(name='red_pillar', type='cylinder', pos='.8 .75 .35', size='.16 .35', rgba='.9 .12 .09 1'),
        dict(name='blue_box', type='box', pos='-1.1 .8 .22', size='.22 .22 .22', rgba='.12 .25 .9 1'),
        dict(name='sound_low', type='box', pos='2.6 -1.4 .3', size='.10 .10 .3', rgba='.8 .8 .25 1'),
        dict(name='sound_high', type='box', pos='-1.6 1.8 .3', size='.10 .10 .3', rgba='.7 .2 .7 1'),
        dict(name='low_step', type='box', pos='1.3 -1 .012', size='.20 .5 .012', rgba='.7 .65 .55 1'),
        # A curb tall enough that the feet must clear it. The proximity rays sit
        # above it, so it is found by touching it, not by seeing it.
        dict(name='curb', type='box', pos='1.15 0 .03', size='.045 .35 .03', rgba='.55 .5 .45 1'),
        dict(name='platform', type='box', pos='1.8 -2.1 .028', size='.55 .3 .028', rgba='.7 .65 .55 1'),
        dict(name='ramp', type='box', pos='.95 -2.1 .018', size='.3 .3 .02', euler='0 -0.06 0', rgba='.6 .6 .6 1'),
        dict(name='passage_a', type='box', pos='-1.8 -.65 .25', size='.65 .07 .25', rgba='.5 .6 .65 1'),
        dict(name='passage_b', type='box', pos='-1.8 -1.45 .25', size='.65 .07 .25', rgba='.5 .6 .65 1'),
    ]
    for prop in props:
        ET.SubElement(world, 'geom', **prop, material='textured')
    # Four eye muscles rest at their zero angle, so the keyframe grows by four
    # entries for the joints and four for the actuators.
    muscles = [('eye_left_yaw', 'eye_left_yaw_joint'), ('eye_left_pitch', 'eye_left_pitch_joint'),
               ('eye_right_yaw', 'eye_right_yaw_joint'), ('eye_right_pitch', 'eye_right_pitch_joint')]
    actuator = robot.find('actuator')
    for name, joint in muscles:
        ET.SubElement(actuator, 'position', name=name, joint=joint, kp='1.2', kv='.05')
    key = robot.find('keyframe/key')
    key.set('qpos', key.get('qpos') + ' 0'*len(muscles))
    key.set('ctrl', key.get('ctrl') + ' 0'*len(muscles))
    ET.indent(robot)
    ET.ElementTree(robot).write(destination, encoding='utf-8', xml_declaration=True)
    return destination


if __name__ == '__main__':
    print(build())
