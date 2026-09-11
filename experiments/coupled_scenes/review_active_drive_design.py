"""Desktop gallery and before/after sheet from actual Blender appearance renders."""
import argparse
import html
import json
import shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    done=json.loads((args.input/'build_complete.json').read_text())
    assert done['complete'] and done['appearance_revision']==3 and not done['physics_ready']
    args.output.mkdir(parents=True,exist_ok=False)
    font='/mnt/c/Windows/Fonts/msyh.ttc'
    title=ImageFont.truetype(font,32);body=ImageFont.truetype(font,22)
    comparison=Image.new('RGB',(1600,2110),'#182121');draw=ImageDraw.Draw(comparison)
    draw.text((25,18),'主动驱动 · 外观改版',font=title,fill='white')
    draw.text((25,68),'左：旧装置    右：新外观样机（空容器；未模拟水、未配套碰撞）',font=body,fill='#c1d7d2')
    overview=Image.new('RGB',(1280,3180),'#182121');overview_draw=ImageDraw.Draw(overview)
    overview_draw.text((25,15),'主动驱动 · 新外观样机 / 未模拟水',font=title,fill='white')
    notes=['带壶嘴的陶瓷壶 + 低矮椭圆接水盆；单侧后置支撑。',
           '薄壁圆形玻璃杯 + 细轴三叶桨；紧凑电机在上方。',
           '细长观察槽 + 端部一体外壳；推杆与滑台收进底座。']
    sections=[]
    for index,key in enumerate(done['cases']):
        source=args.input/key;spec=json.loads((source/'design_spec.json').read_text());case=spec['case']
        dest=args.output/key;dest.mkdir()
        for path in source.iterdir():
            if path.suffix in ('.png','.json','.blend'):shutil.copy2(path,dest/path.name)
        old=args.previous/key/'pose_00.png';shutil.copy2(old,dest/'previous.png')
        y=115+index*660;draw.text((25,y),case['title'],font=title,fill='white')
        for col,path in enumerate((old,dest/'pose_00.png')):
            with Image.open(path) as image:
                comparison.paste(image.convert('RGB').resize((770,578),Image.Resampling.LANCZOS),(15+800*col,y+45))
        draw.text((25,y+627),notes[index],font=body,fill='#c1d7d2')
        oy=65+index*1035
        overview_draw.text((25,oy),case['title'],font=title,fill='white')
        with Image.open(dest/'pose_00.png') as image:overview.paste(image.convert('RGB'),(0,oy+45))
        overview_draw.text((25,oy+1007),notes[index],font=body,fill='#c1d7d2')
        gallery=''.join(f'<figure><a href="{key}/{name}"><img src="{key}/{name}"></a><figcaption>{label}</figcaption></figure>'
            for name,label in [('pose_00.png','初始姿态'),('pose_01.png','动作中段'),('pose_02.png','动作姿态'),('alternate.png','另一视角')])
        sections.append(f'<section><h2>{html.escape(case["title"])}</h2><p>{notes[index]}</p><div class="grid">{gallery}</div>'
            f'<p><a href="{key}/{key}.blend">可编辑 Blender 文件</a> · <a href="{key}/design_spec.json">设计状态说明</a></p></section>')
    comparison.save(args.output/'before_after.png');overview.save(args.output/'overview.png')
    document='''<!doctype html><meta charset="utf-8"><title>主动驱动·外观样机</title>
<style>body{background:#182121;color:#e5eeeb;font:18px sans-serif;max-width:1600px;margin:30px auto;padding:20px}a{color:#9dd8cc}section{margin:30px 0;padding:20px;background:#243330}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}figure{margin:0}img{width:100%}p{line-height:1.6}</style>
<h1>主动驱动 · 外观样机 v3</h1><p><a href="before_after.png">查看新旧对照</a> · <a href="overview.png">新外观总览</a></p>
<p>全部为实际 Blender 模型与渲染，不是生成式概念图。空容器用于确认造型；未运行水体仿真。保存了装置动作和 8 个固定相机。</p>
<p>新壶、圆杯、浅槽的碰撞结构尚未配套，不能套用 v2 的碰撞 USD 或宣称已经可运行。认可外观后再同步修改物理结构。场景贴图仍引用 Y 盘环境资源。</p>'''+''.join(sections)
    (args.output/'review.html').write_text(document,encoding='utf-8')
    (args.output/'说明.txt').write_text('先看 before_after.png（左旧右新）或 review.html。\n新版本为空容器外观样机，未模拟水，未配套新碰撞结构；不要使用旧 v2 碰撞资产。\n子目录有可编辑 Blender 文件、装置动作、8 相机和设计状态。环境贴图仍引用本机 Y 盘。\n',encoding='utf-8')
    print(args.output/'before_after.png')


if __name__=='__main__':main()
