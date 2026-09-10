# 기존 개발 영상의 카메라 움직임 감사

2026-09-10. 깊이 모델을 재실행하거나 임계값을 변경하지 않고 pose 기록만 분석했다.
첫 유효 pose 기준 변위. 저움직임 표시는 장착 고정의 증명이 아니다. 일부 구간에만 유효 pose가 있으면 그 부분만으로 고정을 판정하지 않는다.

| 구간 | 시퀀스 | 클립 | Pose 대응률 | 이동 p95(m) | 회전 p95(deg) | 판정 |
|---|---|---:|---:|---:|---:|---|
| L32 | rgbd_dataset_freiburg3_walking_static | 0 | 93.8% | 0.0145 | 3.802 | unclassified_incomplete_pose |
| L32 | rgbd_dataset_freiburg3_walking_static | 1 | 100.0% | 0.0053 | 1.243 | camera_motion_present |
| L32 | rgbd_bonn_crowd2 | 2 | 100.0% | 0.0606 | 11.427 | camera_motion_present |
| L32 | rgbd_bonn_crowd2 | 3 | 100.0% | 0.0280 | 3.696 | camera_motion_present |
| L32 | rgbd_bonn_person_tracking2 | 4 | 96.9% | 0.1096 | 14.726 | camera_motion_present |
| L32 | rgbd_bonn_person_tracking2 | 5 | 100.0% | 0.0891 | 8.020 | camera_motion_present |
| L32 | rgbd_bonn_static_close_far | 6 | 100.0% | 0.1861 | 6.154 | camera_motion_present |
| L32 | rgbd_bonn_static_close_far | 7 | 100.0% | 0.1401 | 3.914 | camera_motion_present |
| L256 | rgbd_dataset_freiburg3_walking_static | 0 | 100.0% | 0.0311 | 2.656 | camera_motion_present |
| L256 | rgbd_bonn_crowd2 | 1 | 100.0% | 0.2001 | 52.441 | camera_motion_present |
| L256 | rgbd_bonn_person_tracking2 | 2 | 100.0% | 1.0766 | 95.429 | camera_motion_present |
| L256 | rgbd_bonn_static_close_far | 3 | 100.0% | 3.6280 | 19.445 | camera_motion_present |

기존 4개 개발 시퀀스 전체를 고정 카메라 벤치마크로 재명명하지 않는다. 고정 설치가 확인된 별도 장소의 RGB-D 자료가 주 검증에 필요하다.
`work_dirs/qfixed_scope_20260910/audit.json`에 전체 시퀀스 진단, 최대 변위/회전, 입력 hash를 보존했다.
