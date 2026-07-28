ckpt: work_dirs/main_v8/latest.pt
layout: RGB | predicted depth | GT depth
colour: inverse depth, range = GT 1st-99th percentile, shared by
  prediction and GT; prediction median-scaled to GT first.
black in the GT tile = no valid depth. active%% in the filename is
the mean patch-activity over frames 1..N of the clip.
