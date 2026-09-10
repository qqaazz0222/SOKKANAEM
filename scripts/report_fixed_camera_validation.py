"""Publish the data-admission failure and separate RGB-only pilot, without promotion."""
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.acquire_fixed_camera_sbm import ROOT
from scripts.fixed_camera_admission import complete_prefix_members, ur_depth_m
from sokkanaem.protocol import file_record


def main():
    admission=json.loads((ROOT/'admission.json').read_text())
    diagnostic=json.loads((ROOT/'depth_diagnostic.json').read_text())
    bench=json.loads((ROOT/'rgb_only_benchmark/results.json').read_text())
    out=Path('paper/streaming_draft/fixed_camera_validation')
    out.mkdir(exist_ok=False)
    raw=complete_prefix_members('work_dirs/qfixed_public_probe_20260910/ur_fall01_cam1_depth.zip')[0][1]
    source_mm=ur_depth_m(raw)*1000
    rgb=np.array(Image.open(admission['pairs'][0][1]))
    corrected=np.array(Image.open(admission['pairs'][0][2]))
    lower,upper=source_mm[raw>0].min(),source_mm.max()
    flagged=(corrected>0)&((corrected<lower-1)|(corrected>upper+1))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dce2e6')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlecolor':'#263b48',
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,4,figsize=(14.4,3.6),layout='constrained')
    axes[0].imshow(rgb);axes[0].set_title('RGB · frame 1')
    axes[1].imshow(np.ma.masked_where(raw==0,source_mm),cmap=cmap,vmin=0,vmax=4000)
    axes[1].set_title('Original depth · mm\nNative depth coordinates')
    im=axes[2].imshow(np.ma.masked_where(corrected==0,corrected),cmap=cmap,vmin=0,vmax=4000)
    axes[2].set_title('SBM registered values\nMetric-reference admission failed')
    mask=np.full((*flagged.shape,3),[.88,.91,.93]);mask[flagged]=[.89,.32,.20]
    axes[3].imshow(mask);axes[3].set_title(f'Outside original depth range\n{flagged.sum():,} pixels; not model error')
    for ax in axes:ax.axis('off')
    fig.colorbar(im,ax=axes[1:3],shrink=.7,label='Original mm / SBM stored value')
    for ext in ('png','pdf'):fig.savefig(out/f'depth_admission.{ext}',dpi=180)
    plt.close(fig)
    rows=bench['results'];dense=rows['dense']['mean_of_run_means_ms'];mlp=rows['mlp_q50']['mean_of_run_means_ms']
    lines=['# 고정 카메라 검증 진행 결과 — 2026-09-10','',
      '## 결론','',
      '**RGB 기반 실행 검증은 완료했지만, GT 깊이 정확도 검증은 보류했다.** '
      '원본과 수정본의 깊이 범위 불일치를 확인했고, 초기 데이터 승인 실패를 유지한다. '
      '모델·가중치·MLP 임계값은 변경하지 않았으며 새 후보로 승격하지 않았다.','',
      '## 1. 데이터 확보와 고정 촬영 확인','',
      '- 공식 Shadows.zip 630,619,899 bytes 확보. 선택한 fall01cam1 입력 160프레임과 수정 깊이 160프레임의 ID가 1–160으로 연속·일대일 대응한다. 선택 파일은 ZIP CRC와 SHA256을 확인했다.',
      '- URFD 공식 문서에서 camera 1은 천장 고정 카메라이다. SBM은 해당 파생 영상의 동기화 및 2017년 깊이 정렬/스케일 수정 사실을 문서화했다. 파생 영상에는 원 촬영 timestamp가 없어 실제 프레임 간격까지 독립 검증한 것은 아니다.',
      '- 불완전 URFD 원본 ZIP에서 **CRC가 확인되는 완전한 파일만** 진단용으로 읽었다. RGB 5장은 SBM RGB와 픽셀 단위로 일치했다. 원본 ZIP 전체를 복구했다고 주장하지 않는다.',
      '- 7개 전경 주석 사이 6구간의 배경 특징 진단: 각 558–642 inlier, 중앙 변위 0px, 추정 평행이동 최대 0.0324px. 이는 고정 설치 문서를 보조하는 희소 RGB 진단이지 pose GT나 모든 프레임의 무진동 증명이 아니다.',
      '- 기록된 configs/manifests/Q0 설정/정책 실험 protocol에서 해당 데이터 식별자가 발견되지 않았다. DA2 사전학습 중복 여부는 알 수 없다. URFD 원본, 다른 카메라 뷰, SBM 파생본은 같은 이벤트 그룹으로 취급한다. 단일 이벤트는 여러 독립 장소 검증이 아니다.','',
      '출처: [URFD 공식 문서](https://fenix.ur.edu.pl/~mkepski/ds/uf.html), '
      '[SBM 공식 문서](https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset.html).','',
      '## 2. 깊이 정답 승인 실패','',
      '- 원본 camera 1의 공개 변환식 `mm = PNG16 × 3640 / 65535`를 그대로 적용했다. Q0 예측이나 모델 오차에 맞춘 스케일 피팅은 하지 않았다.',
      '- 첫 원본 프레임 유효 깊이는 약 2174–3390mm이다. 대응 수정본에는 값 1부터 시작하는 양수가 있으며, 7898픽셀이 원본 범위 밖이다. 검사한 원본 11프레임 모두에서 이러한 불일치가 관찰됐다.',
      '- 수정본 유효값이 원본에서 나온 반올림 깊이값의 ±1mm 안에 있는 비율은 96.35–98.88%로, 데이터 검사에 미리 정한 99% 기준을 충족하지 못했다. 이는 모델 품질 허용치와 별개의 데이터 진단이다.',
      '- 무효 깊이 주변을 1/2/3/4/8픽셀 제외하는 진단에서도 범위 밖 값이 일부 남았다. 원인을 보간·정렬로 단정하거나 임의 마스크로 해결됐다고 처리하지 않았다. 원본과 수정본은 좌표계도 달라 직접 픽셀 오차로 정렬 품질을 판정할 수 없다.',
      '- 따라서 수정본 전체를 그대로 센서 GT로 사용한 AbsRel·경계·선명도 수치나 품질 gate 결과는 생성하지 않았다. 단순 단위 오류라고 단정한 것도 아니다. 유효 영역 정의와 정렬 처리 근거를 추가 확보해야 한다.','',
      '![깊이 데이터 승인 진단](fixed_camera_validation/depth_admission.png)','',
      '첫 프레임의 원본 센서 좌표와 SBM 등록 좌표를 구분해 표시했다. 붉은 영역은 원본 범위 밖의 저장값이며, **모델 예측 오류 이미지가 아니다**.','',
      '## 3. 별도 RGB-only 실행 검증','',
      '깊이 정답을 쓰지 않는 별도 protocol을 예측 전에 기록했다. 전체 160프레임을 한 스트림으로 처리하고, 매 반복 시작에만 상태를 초기화했다. '
      '3회 반복, 동일 고정 순서, shared RTX4090, 내부 Q0 518/출력 256/DIS 128. 입력 파일 읽기와 초기 H2D 제외, 내부 CPU 흐름 및 전송 포함. '
      '30fps 실시간 입력이나 Nano/Pi end-to-end 검증이 아니다.','',
      '| 방식 | 평균 ms/프레임 | 반복별 p95 범위 ms | 전체 Q0 호출 | 전체 Q0 대비 출력 상대차 평균 |',
      '|---|---:|---:|---:|---:|']
    names={'dense':'Q0 dense','hold_k2':'단순 hold K2','flow_k2':'flow K2','mlp_q50':'SOKKANAEM-Refresh MLP'}
    for name,row in rows.items():
        tail=[r['p95_ms'] for r in row['runs']]
        lines.append(f'| {names[name]} | {row["mean_of_run_means_ms"]:.3f} | {min(tail):.3f}–{max(tail):.3f} | {row["runs"][0]["full_calls"]}/160 | {100*row["reference_relative_difference_mean"]:.3f}% |')
    lines += ['',
      f'- MLP 평균은 dense 대비 {dense/mlp:.3f}×이며, flow K2보다 {100*(1-mlp/rows["flow_k2"]["mean_of_run_means_ms"]):.2f}% 짧다. '
      '하지만 MLP의 전체 Q0 출력 상대차는 K2보다 크고, p95도 dense보다 크다. 평균 가속을 품질 보존이나 최악 지연 개선으로 바꾸어 주장하지 않는다.',
      '- 출력 상대차는 `mean(abs(pred−Q0)/Q0)`이다. **실제 깊이 정답에 대한 정확도가 아니며 기존 GT 기반 1% gate와 비교할 수 없다.**',
      '- 희소 전경 주석 6프레임의 Q0 대비 차이는 MLP 3.147%, flow K2 6.413%, hold 7.892%이다. '
      '단, 주석 시점과 갱신 주기가 불균등하게 겹치므로 이를 전경 정확도 우위로 해석하지 않는다. 전경은 멈춘 물체도 포함하며 순간 움직임/가림 해제 GT가 아니다.',
      f'- 전체 갱신 {bench["exact_refresh_comparisons"]}회가 Q0와 정확히 일치하고, 반복 출력 {bench["exact_repeated_frames"]}프레임이 정확히 일치했다. 관련 회귀 테스트 107개 통과.',
      '', '## 4. 대안 점검 및 남은 순서','',
      '- 같은 공식 묶음의 genSeq2(300프레임), shadows1(260), shadows2(250) 원시 PNG 샘플은 16비트이며 양수 범위가 각각 2392–4296, 758–2690, 761–2609이다. 범위만으로 mm를 확정하지 않는다. 이 영상들은 모델 추론하지 않았다.',
      '- Camplani–Salgado 원 논문은 정적인 장치로 수집한 실내 영상이라고 설명한다. 그러나 논문 실험은 깊이를 8비트로 변환해 쓰므로, 현재 SBM 16비트 파일의 단위·정렬 provenance를 이것만으로 확정할 수 없다. 원 데이터 웹 경로는 현재 Article not found로 응답했다. [저자 기관 원문](https://oa.upm.es/37450/).',
      '- 다음 우선순위는 URFD 원 깊이와 등록 보정/유효 마스크의 문서화 또는 다른 고정 카메라 원본의 metric 정렬 근거 확보이다. 데이터 승인 전 GT 점수 계산은 계속 보류한다.',
      '- 이후 새로운 장소/이벤트 그룹을 고정해 4개 제어군의 GT 정확도·경계·장면별 지표를 평가한다. 현재 160프레임을 반복하거나 다른 이벤트와 연결하여 L256을 만들지 않는다.',
      '- 신뢰할 GT를 확보하더라도 여러 독립 장소, 장시간 연속 입력, 목표 장치 Nano B01 5W/10W 및 Pi4 측정은 별도 미완료 항목이다.',
      '', '## 재현 자료','',
      '`work_dirs/qfixed_validation_20260910/`: acquisition_protocol.json, acquired.json, admission.json, depth_diagnostic.json, rgb_only_benchmark/protocol.json 및 results.json.',
      '원본 자료와 최초 승인 실패는 보존했다. 원본 dataset 파일은 paper_work에 재배포하지 않는다.']
    report=Path('paper/streaming_draft/FIXED_CAMERA_VALIDATION.md')
    with report.open('x') as f:f.write('\n'.join(lines)+'\n')
    provenance={'source':file_record(__file__), 'inputs':[file_record(ROOT/'admission.json'),file_record(ROOT/'depth_diagnostic.json'),
                file_record(ROOT/'rgb_only_benchmark/results.json')], 'report':file_record(report),
                'figures':[file_record(out/f'depth_admission.{e}') for e in ('png','pdf')],
                'scope':'data-only figure, RGB-only timing; no model GT scores'}
    (ROOT/'report_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
