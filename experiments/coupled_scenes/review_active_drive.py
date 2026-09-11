"""Compose existing apparatus renders into a desktop gallery; no new rendering."""
import argparse
import html
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    complete=json.loads((args.input/'build_complete.json').read_text())
    assert complete['complete'] and len(complete['cases'])==3
    args.output.mkdir(parents=True,exist_ok=False)
    font='/mnt/c/Windows/Fonts/msyh.ttc'
    title=ImageFont.truetype(font,32);body=ImageFont.truetype(font,22)
    canvas=Image.new('RGB',(1600,2170),'#172129');draw=ImageDraw.Draw(canvas)
    draw.text((25,18),'主动驱动 · 倾倒 / 搅拌 / 活塞',font=title,fill='white')
    draw.text((25,70),'仅装置布局与动作预览；左图水块只表示初始填水区域，右图不显示占位水。',font=body,fill='#bcd8e2')
    notes=['绕上方容器出水沿翻转 105°，停留后回正；下方接水槽初始为空。',
           '浸入式十字桨：平滑起转 → 保持 1.8 rad/s → 减速停止。',
           '推板前方预填水，向前推 45 cm 后保持；不回抽。']
    sections=[]
    for index,key in enumerate(complete['cases']):
        source=args.input/key;spec=json.loads((source/'scene_spec.json').read_text());case=spec['case']
        dest=args.output/key;dest.mkdir()
        for path in source.iterdir():
            if path.suffix in ('.png','.json','.usda','.blend'): shutil.copy2(path,dest/path.name)
        y=125+index*675
        draw.text((25,y),case['title'],font=title,fill='white')
        for col,name in enumerate(('pose_00.png','pose_02.png')):
            with Image.open(dest/name) as image:
                canvas.paste(image.convert('RGB').resize((770,578),Image.Resampling.LANCZOS),(15+col*800,y+48))
            time=case['review_times_s'][0 if col==0 else 2]
            draw.text((30+col*800,y+54),f't = {time:g} s'+(' · 初始填水示意' if col==0 else ' · 仅装置'),font=body,fill='white',stroke_width=2,stroke_fill='#172129')
        draw.text((25,y+632),notes[index],font=body,fill='#bed0d7')
        images=''.join(f'<figure><img src="{key}/pose_{i:02d}.png"><figcaption>t = {t:g} s'+
            ('，初始填水示意' if i==0 else '，仅装置，无模拟水')+'</figcaption></figure>' for i,t in enumerate(case['review_times_s']))
        sections.append(f'<section><h2>{html.escape(case["title"])}</h2><p>{html.escape(case["observation"])}</p><div class="poses">{images}</div>'
            f'<p><a href="{key}/alternate.png">另一视角</a> · <a href="{key}/{key}.blend">可编辑 Blender 场景</a> · '
            f'<a href="{key}/scene_spec.json">场景说明</a> · <a href="{key}/colliders.usda">碰撞几何 USD</a></p></section>')
    canvas.save(args.output/'overview.png')
    document='''<!doctype html><meta charset="utf-8"><title>主动驱动场景</title>
<style>body{background:#172129;color:#e3edf0;font:18px sans-serif;max-width:1600px;margin:30px auto;padding:20px}a{color:#8adddf}.poses{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}section{padding:20px;background:#243039;margin:24px 0}p{line-height:1.6}</style>
<h1>主动驱动场景 · 布局与动作</h1><p>三套装置位于现有 warehouse 环境内，每套配 8 个固定相机、10 秒装置动作。未启动水体仿真。</p>
<p>初始帧的水块仅标示填水区域；动作开始后的预览隐藏占位水，不表示水已消失或已完成倾倒。Blender 时间轴保存完整装置动作。</p>
<p>场景仍引用 Y 盘环境贴图，不是跨机器自包含资源包。原生流体运行适配尚未接通，不能直接交给旧倾倒入口运行。</p>'''+''.join(sections)
    (args.output/'review.html').write_text(document,encoding='utf-8')
    (args.output/'说明.txt').write_text('先打开 overview.png 或 review.html。\n每个子目录有可编辑 .blend、碰撞 USD、8 视角相机与解析动作采样。\n所有水块均为初始填水设计参考，未进行流体仿真；动作期间占位水隐藏。\n在 Blender 打开场景后播放时间轴可查看 10 秒装置动作。\n场景贴图仍引用 Y 盘既有资源。\n',encoding='utf-8')
    print(args.output/'overview.png')


if __name__=='__main__': main()
