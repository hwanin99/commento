import pytest
import numpy as np
from processing import basic_depth_map, advanced_depth_map

def test_basic_depth_map():
    img_path = './sample_3.jpg'
    generator = basic_depth_map(img_path)
    image = generator.load_image()
    depth_map = generator.generate_depth_map()
    generator.show_image()  # GUI 확인용, 자동 테스트에서는 생략 가능

    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."

def test_advanced_depth_map():
    img_path = './sample_3.jpg'
    generator = advanced_depth_map(img_path)
    image = generator.load_image()
    depth_map = generator.generate_depth_map()
    points = generator.point_cloud()
    generator.show_image()  # GUI 확인용, 자동 테스트에서는 생략 가능

    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."
    assert points.shape[2] == 3, "3D 포인트가 (X, Y, Z) 형태가 아닙니다."

if __name__ == "__main__":
    pytest.main()
