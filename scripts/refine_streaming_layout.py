"""Code-native layout refinement; scientific predictions remain unchanged."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write
out=Path('paper/streaming_draft');prov=json.loads((out/'visual_provenance.json').read_text())
for r in prov['predictions']+[prov['data']]:assert file_record(r['path'])==r
fig,ax=plt.subplots(figsize=(12,4.8));ax.set(xlim=(0,12),ylim=(-.8,4.1));ax.axis('off')
def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.12',facecolor=color,edgecolor='#b9c8d6'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11,color='#22354d')
def arrow(a,b,label=None):
    ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':'#526477','lw':1.6})
    if label:ax.text((a[0]+b[0])/2-.12,(a[1]+b[1])/2+.13,label,ha='center',fontsize=9,color='#22354d')
box(.2,1.4,1.3,1,'Current RGB\nframe','#e9f0f7');box(2,1.4,2.4,1,'RGB change + age\nMLP decision','#dcebf4')
box(5.2,2.6,2.5,.85,'Frozen Q0\nfull inference','#e0ece6');box(5.2,.35,2.5,.85,'DIS flow + nearest\nprevious-depth warp','#f2e9da')
box(8.4,1.4,1.35,1,'Depth\noutput','#e9f0f7');box(10.25,1.4,1.5,1,'Cache\nupdate','#e9f0f7')
arrow((1.5,1.9),(2,1.9));arrow((4.4,2.1),(5.2,2.95),'Refresh');arrow((4.4,1.6),(5.2,.85),'Reuse')
arrow((7.7,2.95),(8.4,2.1));arrow((7.7,.85),(8.4,1.6));arrow((9.75,1.9),(10.25,1.9))
ax.plot([11,11,3.2],[2.4,3.85,3.85],ls='--',color='#526477',lw=1.2)
ax.annotate('',(3.2,2.4),(3.2,3.85),arrowprops={'arrowstyle':'->','linestyle':'--','color':'#526477'})
ax.text(8.8,3.63,'Last-refresh RGB anchor / age',ha='center',fontsize=9,color='#526477')
ax.plot([11,11,6.45],[1.4,-.15,-.15],ls='--',color='#526477',lw=1.2)
ax.annotate('',(6.45,.35),(6.45,-.15),arrowprops={'arrowstyle':'->','linestyle':'--','color':'#526477'})
ax.text(9,-.42,'Previous depth / grayscale',ha='center',fontsize=9,color='#526477')
ax.text(.2,-.72,'Gap 1: reuse     |     Gaps 2–3: learned decision     |     Gap 4: refresh',fontsize=10,color='#22354d')
for ext in ('png','pdf'):fig.savefig(out/'figures'/('architecture.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
prov['layout_revision']=file_record(__file__);prov['artifacts']=[file_record(p) for p in sorted((out/'figures').iterdir())]
write(out/'visual_provenance.json',prov)
