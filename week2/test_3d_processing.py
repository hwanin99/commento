import numpy as np
import cv2
import pytest
import open3d as o3d


######### 기본적인 Depth Map 생성 코드 (OpenCV 활용) #########
class basic_depth_map:
    def __init__(self, img_path):
        self.img_path = img_path
        self.image = None
        self.gray = None
        self.depth_map = None

    def load_image(self):
        # 이미지 로드
        self.image = cv2.imread(self.img_path)
        if self.image is None:
            raise ValueError("입력된 이미지가 없습니다.")
        # 그레이스케일 변환
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        return self.image

    def generate_depth_map(self):
        # 깊이 맵 생성 (가상의 깊이 적용)
        self.depth_map = cv2.applyColorMap(self.gray, cv2.COLORMAP_JET)
        
        return self.depth_map
    
    def show_image(self):
        # 결과 출력
        cv2.imshow('Original Image', self.image)
        cv2.imshow('Depth Map', self.depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class advanced_depth_map:
    def __init__(self, img_path):
        self.img_path = img_path
        self.image = None
        self.gray = None
        self.depth_map = None
        self.points_3d = None
        self.colors = None

    def load_image(self):
        # 이미지 로드
        self.image = cv2.imread(self.img_path)
        if self.image is None:
            raise ValueError("입력된 이미지가 없습니다.")
        # 그레이스케일 변환
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        return self.image

    def generate_depth_map(self):
            # Depth Map 생성
            self.depth_map = cv2.applyColorMap(self.gray, cv2.COLORMAP_JET)
            
            return self.depth_map

    def point_cloud(self):
            # 3D 포인트 클라우드 변환
            h, w = self.depth_map.shape[:2]
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            Z = self.gray.astype(np.float32) # Depth 값을 Z 축으로 사용

            # 3D 좌표 생성
            self.points_3d = np.dstack((X, Y, Z))
            
            return self.points_3d

    def conversion(self):
        points_3d = self.points_3d
        # 좌표계 보정 (Y,Z 방향 반대)
        X = self.points_3d[:, :, 0]
        Y = -self.points_3d[:, :, 1] # -Y
        Z = -self.points_3d[:, :, 2] # -Z

        # reshape (N,3)형태
        converted_points_3d = np.dstack((X,Y,Z)).reshape(-1, 3)
        self.colors = self.depth_map.reshape(-1, 3) / 255.0
        
        return converted_points_3d, self.colors

    def show_image(self):
        # OpenCV
        cv2.imshow('Original Image', self.image)
        cv2.imshow('Depth Map', self.depth_map)

        # Open3D
        converted_points_3d, colors = self.conversion()

        p3d = o3d.geometry.PointCloud()

        p3d.points = o3d.utility.Vector3dVector(converted_points_3d) #(N,3) 형태를 받음.
        p3d.colors = o3d.utility.Vector3dVector(colors)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])

        o3d.visualization.draw_geometries([p3d, axes],
                                        window_name='3D Point Cloud',
                                        width=960, height=720,
                                        point_show_normal=False)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


############ pytest ############
def test_basic_depth_map():
    # img_path = 'Your Image path'
    img_path = './sample_3.jpg'
    generator = basic_depth_map(img_path)
    image = generator.load_image()
    depth_map = generator.generate_depth_map()
    generator.show_image()

    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."


def test_advanced_depth_map():
    # img_path = 'Your Image path'
    img_path = './sample_3.jpg'
    generator = advanced_depth_map(img_path)
    image = generator.load_image()
    depth_map = generator.generate_depth_map()
    points = generator.point_cloud()
    generator.show_image()

    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."
    assert points.shape[2] == 3, "3D 포인트가 (X, Y, Z) 형태가 아닙니다."


############ pytest ############
if __name__ == "__main__":
    pytest.main()