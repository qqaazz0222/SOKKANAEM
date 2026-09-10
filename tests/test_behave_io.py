import shutil
import subprocess
import numpy as np
import pytest
from sokkanaem.behave_io import (unpack_depth_yuv444, depth_metres,
                                iter_depth_video, associate_timestamps)


def pack_fixture(a):
    # Test fixture implements the format from its two bytes, not a production encoder.
    hi = (a//256).astype(np.uint8)
    lo = (a%256).astype(np.uint8)
    folded = np.where(hi%2, 255-lo, lo).astype(np.uint8)
    return np.stack([folded, np.zeros_like(lo), hi])


def test_all_uint16_values_roundtrip():
    a = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    np.testing.assert_array_equal(unpack_depth_yuv444(pack_fixture(a)), a)


def test_metric_units_and_invalid_zero():
    a = depth_metres(np.array([[0, 1000, 65535]], np.uint16))
    np.testing.assert_allclose(a, [[0, 1, 65.535]])
    with pytest.raises(ValueError): depth_metres(np.array([[1000.0]]))


def test_reject_preview_layout():
    with pytest.raises(ValueError): unpack_depth_yuv444(np.zeros((12, 16, 3), np.uint8))


def test_timestamps_keep_unmatched_rgb_without_reusing_depth():
    r = associate_timestamps([0, 10, 20, 30], [1, 29], 12)
    assert len(r['frames']) == 4 and r['matched'] == 2
    assert [v['depth_index'] for v in r['frames']] == [0, None, None, 1]


def test_timestamp_ties_choose_earlier():
    r = associate_timestamps([5, 25], [0, 10, 20, 30], 5)
    assert [v['depth_index'] for v in r['frames']] == [0, 2]


@pytest.mark.parametrize('values', [[0, 0], [2, 1], [0, float('nan')], [0]])
def test_timestamp_invalid_order(values):
    with pytest.raises(ValueError): associate_timestamps(values, [0, 1], 1)


def test_tolerance_is_not_auto_relaxed():
    assert associate_timestamps([0, 100], [20, 120], 5)['matched'] == 0
    with pytest.raises(ValueError): associate_timestamps([0, 1], [0, 1], -1)


@pytest.fixture
def encoded_video(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    arrays = np.random.default_rng(504).integers(0, 65536, (5, 16, 32), dtype=np.uint16)
    path = tmp_path/'fixture.mp4'
    packed = b''.join(pack_fixture(a).tobytes() for a in arrays)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv444p',
                    '-s', '32x16', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '0', '-pix_fmt', 'yuv444p', str(path)],
                   input=packed, capture_output=True, check=True, timeout=30)
    return path, arrays


def test_actual_mp4_frame_order_and_lossless_decode(encoded_video):
    path, arrays = encoded_video
    decoded = np.stack(list(iter_depth_video(path, expected_count=5)))
    np.testing.assert_array_equal(decoded, arrays)


def test_actual_mp4_timestamp_count_mismatch(encoded_video):
    path, _ = encoded_video
    with pytest.raises(ValueError, match='frame count'):
        list(iter_depth_video(path, expected_count=6))
