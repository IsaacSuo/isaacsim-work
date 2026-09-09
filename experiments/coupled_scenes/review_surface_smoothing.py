"""Compare matched actual renders and identical pixel crops (WSL/Pillow)."""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',30)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',24)
    canvas=Image.new('RGB',(2400,1260),'#172129')
    draw=ImageDraw.Draw(canvas)
    draw.text((20,15),'同一份 8 秒末态 · 只改变网格平滑次数',font=font,fill='white')
    draw.text((20,60),'上排：相同近景视角   下排：同一区域等比例放大   /   未重跑仿真',font=small,fill='#bad4db')
    reports=[]
    for column,iterations in enumerate((0,15,25)):
        suffix='' if iterations==0 else f'_smooth{iterations}'
        render=args.source/f'render_t8s{suffix}'
        surface=args.source/('snapshot_surface' if iterations==0 else f'snapshot_surface_smooth{iterations}')
        metadata=json.loads((surface/'snapshot_surface.json').read_text())
        reports.append(metadata)
        assert metadata['mesh_smoothing_iters']==iterations
        with Image.open(render/'surface_close.png') as picture:
            assert picture.size==(1600,1200)
            canvas.paste(picture.convert('RGB').resize((780,585),Image.Resampling.LANCZOS),(column*800+10,145))
            # Same 640x360 source region: surface centre, avoids glass rim.
            canvas.paste(picture.crop((460,440,1100,800)).convert('RGB').resize((780,439),Image.Resampling.LANCZOS),(column*800+10,785))
        draw.text((column*800+20,103),f'{iterations} 次平滑',font=font,fill='white')
        draw.text((column*800+20,743),'相同水面区域放大',font=small,fill='#bad4db')
        for view,label in [('overview','整体'),('surface_close','近景')]:
            shutil.copy2(render/f'{view}.png',args.output/f'{iterations}次_{label}.png')
    for key in ('source_sha256','particle_count','smoothing_length','cube_size','surface_threshold','normal_smoothing_iters'):
        assert len({r[key] for r in reports})==1,key
    canvas.save(args.output/'平滑对比.png')
    (args.output/'comparison.json').write_text(json.dumps(dict(
        iterations=[0,15,25],source_sha256=reports[0]['source_sha256'],
        crop_pixels=[460,440,1100,800],physics_unchanged=True,
        note='Static appearance comparison only; temporal stability not tested.'),indent=2),encoding='utf-8')
    print(args.output/'平滑对比.png')


if __name__=='__main__':
    main()
