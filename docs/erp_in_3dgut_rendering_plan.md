# 3D GUT 전용 ERP 카메라/렌더링 통합 계획

본 문서는 3D GUT(Unscented Transform 기반 래스터 경로)에 한해 Equirectangular Projection(ERP) 카메라를 지원하기 위한 구체적 수정 지점과 구현 순서를 정리합니다. 3D GRT(레이 트레이싱 경로)는 본 범위에서 제외합니다.

## 목표
- 3D GUT의 UT 기반 프로젝트/래스터 파이프라인에서 ERP 카메라를 1급 시민으로 지원
- 파이썬(Renderer/데이터셋)에서 ERP 광선(rays_o, rays_d) 생성 가능
- CUDA/UT 측 projectPoint 경로가 ERP를 인지해 올바른 이미지 좌표에 입사 시그마 포인트를 투영

## 현재 파이프라인 요약(3D GUT)
- 파이썬 진입점: `threedgut_tracer/tracer.py`
  - `Tracer.__create_camera_parameters(...)`에서 카메라 모델을 C++ 플러그인(`lib3dgut_cc`)에 전달
  - 현재 지원: OpenCV Pinhole, OpenCV Fisheye
- C++/CUDA 핵심
  - 카메라 모델 정의: `threedgut_tracer/include/3dgut/sensors/cameraModels.h`
  - 투영 로직: `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh`의 `projectPoint(...)`
  - UT 기반 스크린 공간 프로젝터: `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutProjector.cuh`
  - PyTorch 바인딩: `threedgut_tracer/bindings.cpp`

## 진행 상황 (2025-10-01)
- C++/CUDA 구현 완료
  - `threedgut_tracer/include/3dgut/sensors/cameraModels.h`: `EquirectangularModel` 추가, `EquirectangularProjectionParameters{}` 및 `union` 멤버 추가
  - `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh`: ERP `projectPoint(...)` 오버로드 구현, `TSensorModel` 스위치 분기 추가, seam ε-클램프 적용
- PyBind 바인딩 추가
  - `threedgut_tracer/bindings.cpp`: `fromEquirectangularCameraModelParameters(...)` 등록
  - 테스트용 진입점 `_test_project_points(...)` 추가(배치 방향 투영 검증)
- 테스트
  - `tests/test_erp_projection.py`: +Z/+X/−X/+Y/−Y/−Z(seam) 투영 좌표 검증
  - 실행: `conda run -n 3dgrut python tests/test_erp_projection.py`
  - 결과: “ERP projection test passed.”
- 주의사항
  - 컴파일 타임 제약: `GAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID=false` 필요(`threedgut_tracer/include/3dgut/threedgut.cuh`)
- Python 구현 완료
  - 데이터클래스: `threedgrut/datasets/camera_models.py`에 `EquirectangularCameraModelParameters` 추가
  - 레이 유틸: `threedgrut/datasets/utils.py`에 `equirectangular_camera_rays(w,h,ray_jitter=None)` 추가
  - Batch 필드: `threedgrut/datasets/protocols.py`에 `intrinsics_EquirectangularCameraModelParameters` 추가
  - 트레이서 분기: `threedgut_tracer/tracer.py`에 ERP 분기 및 `_3dgut_plugin.fromEquirectangularCameraModelParameters(...)` 연결
- 구성/연계
  - Hydra 렌더 설정에 카메라 선택 키 도입 완료: `configs/render/3dgut.yaml` (`render.camera.model`, `render.camera.yaw_pitch_roll`)
  - COLMAP 데이터셋 경로에서 ERP 강제 오버라이드 분기 구현 완료: 팩토리(`threedgrut/datasets/__init__.py`)→`ColmapDataset`(`threedgrut/datasets/dataset_colmap.py`)
- 남은 작업
  - 단일/소수 프레임 오프라인 렌더에서 전방 클램프 및 타이밍 수집 확인, 성능/회귀 점검

## 필요한 변경 사항

### 1) CUDA/C++: ERP 카메라 모델 정의 및 투영 추가
1. `threedgut_tracer/include/3dgut/sensors/cameraModels.h`
   - `CameraModelParameters::ModelType`에 `EquirectangularModel` 추가
   - `struct EquirectangularProjectionParameters {}` 추가(최소 구현은 필드 없이 시작 가능). 필요 시 후속 PR에서 크롭/왜곡 파라미터 확장
   - `union`에 `EquirectangularProjectionParameters` 멤버 추가

2. `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh`
   - 아래 형태의 오버로드 추가
     - `projectPoint(const EquirectangularProjectionParameters&, const tcnn::ivec2& resolution, const tcnn::vec3& position, float tolerance, tcnn::vec2& projected)`
   - 구현 요지(카메라 좌표계에서):
     - `d = normalize(position)`; 유효성: `position` z-forward 규약 유지, 영벡터/NaN 방지
     - `phi  = atan2(d.x, d.z)`  // [-π, π]
     - `theta = asin(clamp(d.y, -1, 1))`  // [-π/2, π/2]
     - `u = (phi + π) / (2π) * W`, `v = (0.5 - theta/π) * H`  // v는 위(+y) 기준 위에서 아래로 증가
     - `withinResolution(resolution, tolerance, projected)`로 유효성 체크 및 반환
   - `projectPoint(const TSensorModel& ...)` 스위치에 `EquirectangularModel` 분기 추가

3. `threedgut_tracer/bindings.cpp`
   - 바인딩 함수 추가:
     - `fromEquirectangularCameraModelParameters(std::array<uint64_t,2> resolution, threedgut::TSensorModel::ShutterType shutter_type)`
     - 구현: `CameraModelParameters params; params.shutterType = ...; params.modelType = EquirectangularModel;` (필요 시 파라미터 구조체 채움)
   - `PYBIND11_MODULE`에 위 함수 등록

4. 렌더러 측 영향
   - `gutProjector.cuh`는 최종적으로 `projectPoint(...)`를 통해 σ-포인트를 이미지 평면으로 투영하므로 별도 변경 불요
   - 롤링셔터는 기존 `relativeShutterTime(...)`가 픽셀 위치 기반으로 동작(기본: Global)하므로 호환

### 2) Python: ERP 데이터 구조와 광선 생성 지원
5. `threedgrut/datasets/camera_models.py`
   - 새 데이터클래스 추가:
     - `@dataclass class EquirectangularCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin): pass`
     - 초기에는 `resolution`, `shutter_type`만 사용(필요 시 후속 필드 확장)

6. ERP 광선(ray) 생성 유틸 추가(카메라 좌표계 기준)
   - 위치: `threedgrut/datasets/utils.py` 또는 전용 파일(예: `threedgrut/datasets/equirect.py`)
   - 시그니처 예시: `generate_equirect_rays(width, height, yaw=0.0, pitch=0.0, roll=0.0) -> (rays_o, rays_d)`
     - 각 픽셀 `(i,j)`에 대해 `(u=i+0.5, v=j+0.5)` → `(phi, theta)`로 변환
     - `phi = 2π*(u/W) - π`, `theta = π*(0.5 - v/H)`
     - `d_cam = [cos(theta)*sin(phi), sin(theta), cos(theta)*cos(phi)]`
     - 오일러(yaw,pitch,roll) 적용(옵션), `rays_o = 0`, `rays_d = d_cam`

7. GUT 트레이서 파이썬 진입점 확장
   - `threedgut_tracer/tracer.py::Tracer.__create_camera_parameters(...)`
     - 분기 추가:
       ```python
       elif (K := gpu_batch.intrinsics_EquirectangularCameraModelParameters) is not None:
           camera_model_parameters = _3dgut_plugin.fromEquirectangularCameraModelParameters(
               resolution=K["resolution"],
               shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
           )
           return camera_model_parameters, pose_model.get_sensor_pose()
       ```

8. 데이터셋/렌더 경로 연결(추가 옵션)
   - 오프라인 렌더(평가)에서 ERP 카메라로 강제 오버라이드할 수 있도록 설정 키는 이미 존재합니다:
     - `configs/render/3dgut.yaml` 예시
       ```yaml
       camera:
         model: equirectangular  # pinhole|fisheye|equirectangular
         yaw_pitch_roll: [0.0, 0.0, 0.0]
       ```
   - 적용 위치 선택지
     1) `threedgrut/render.Renderer.render_all(...)`에서 `gpu_batch` 후처리로 ERP 광선/내부파라미터로 대체
     2) `threedgrut/datasets/dataset_colmap.py::get_gpu_batch_with_intrinsics(...)`에서 구성에 따라 ERP로 대체
   - 대체 시 `gpu_batch`에 다음 키 제공
     - `rays_ori`, `rays_dir`(ERP 광선)
     - `intrinsics_EquirectangularCameraModelParameters`(데이터클래스 `to_dict()`)

## 좌표/규약 주의사항
- ERP 경로는 카메라 좌표계를 `z`-forward, `x`-right, `y`-up으로 가정하며, `position.z > 0`을 전방으로 간주합니다.
- 일부 주석/문서에 `right down front` 표기가 남아 있을 수 있으므로 혼선을 줄이기 위해 차기 PR에서 일괄 정정 예정입니다.
- ERP `v` 정의는 북극(위쪽)이 작은 v, 남극(아래쪽)이 큰 v가 되도록 `v = (0.5 - theta/π) * H` 채택
- σ-포인트 투영 실패율을 낮추기 위해 `ut_in_image_margin_factor`는 기본값 유지 권장(`configs/render/3dgut.yaml`)

## 테스트 체크리스트
- 단위 방향 벡터 테스트(센터/4분면/극점)에서 투영 좌표가 직관적 범위로 매핑되는지 확인
- 좌우 경계(φ=±π)에서 wrap-around가 자연스럽게 처리되는지(타일링/경계 조건에서 외형적 seam 없음)
- 롤링셔터 Global 기본에서 정상 이미지 생성 후, rolling 분기에서도 치명 오류 없음을 확인
- 오프라인 렌더 1~2프레임으로 회귀 테스트(성능: 해상도 ↑ 시 타일링 옵션/다운샘플 제안)

## 구현 순서 제안
1) C++/CUDA: `cameraModels.h` + `cameraProjections.cuh` + `bindings.cpp`에 ERP 추가
2) Python: `EquirectangularCameraModelParameters` 데이터클래스 + ERP raygen 유틸 추가
3) `Tracer.__create_camera_parameters`에 ERP 분기 추가
4) 구성 키(옵션)로 ERP 강제 렌더를 연결(렌더러 혹은 데이터셋 경로 중 택1)
5) 오프라인 렌더로 시각/성능 검증 후 PR

## 참고 파일(수정 대상 요약)
- C++/CUDA
  - `threedgut_tracer/include/3dgut/sensors/cameraModels.h`
  - `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh`
  - `threedgut_tracer/bindings.cpp`
- Python
  - `threedgut_tracer/tracer.py:__create_camera_parameters`
  - `threedgrut/datasets/camera_models.py`
  - `threedgrut/datasets/utils.py` 또는 신규 `threedgrut/datasets/equirect.py`
  - (옵션) `threedgrut/render.py` 또는 `threedgrut/datasets/dataset_colmap.py`에 ERP 오버라이드

본 계획대로 진행 시, 3D GUT 경로에서만 ERP를 깔끔히 활성화할 수 있으며, 3D GRT와의 혼선을 제거합니다. 이후 필요 시 ERP 전용 파라미터(수평/수직 크롭, yaw/pitch/roll 오프셋 등)를 점진적으로 확장 가능하며, 기존 UT 파이프라인의 안정성을 유지합니다.


## TODO 체크리스트 (실행 계획)

아래 항목들은 실제 구현에 바로 착수할 수 있도록 파일/함수 기준으로 정리한 작업 목록입니다. 각 항목은 “수용 기준(AC)”과 함께 제공되며, PR 분할 시에도 그대로 사용 가능합니다.

### C++/CUDA (카메라 모델/투영/바인딩)
- [x] `threedgut_tracer/include/3dgut/sensors/cameraModels.h`
  - [x] `CameraModelParameters::ModelType`에 `EquirectangularModel` 추가
  - [x] `struct EquirectangularProjectionParameters {}` 추가(초기 빈 구조체)
  - [x] `union`에 `EquirectangularProjectionParameters` 멤버 추가 및 기본 초기화
  - AC: 새 모델 타입이 컴파일되고 기본 생성 시 UB가 없어야 함

- [x] `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh`
  - [x] 시그니처 추가: `projectPoint(const EquirectangularProjectionParameters&, const tcnn::ivec2&, const tcnn::vec3&, float, tcnn::vec2&)`
  - [x] 구현: `position` 정규화 → `phi = atan2(d.x, d.z)`, `theta = asin(clamp(d.y))` → `(u,v)` 매핑
  - [x] v축(Down-카메라 규약) 정합: 위(−Y)가 작은 v, 아래(+Y)가 큰 v 되도록 구현
  - [x] φ=±π 경계 클램프 또는 정규화로 wrap seam OOB 방지(`u = min(max(u,0), W-ε)` 등)
  - [x] `projectPoint(const TSensorModel& ...)` 스위치에 `EquirectangularModel` 분기 추가
  - AC: Pinhole/Fisheye 기존 경로에 영향 없고, ERP 입력 시 유효 좌표 반환 및 `withinResolution` 충족

- [x] `threedgut_tracer/bindings.cpp`
  - [x] 함수 추가: `fromEquirectangularCameraModelParameters(std::array<uint64_t,2> resolution, threedgut::TSensorModel::ShutterType shutter_type)`
  - [x] `params.modelType = EquirectangularModel; params.shutterType = ...;` 설정
  - [x] `PYBIND11_MODULE`에 바인딩 등록
  - AC: 파이썬에서 호출 시 `threedgut::CameraModelParameters`를 반환하고 모듈 로드/트레이스에 성공

### Python (데이터클래스/Batch/레이 생성/트레이서)
- [x] `threedgrut/datasets/camera_models.py`
  - [x] `@dataclass class EquirectangularCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin)` 추가(필드: `resolution`, `shutter_type`)
  - AC: `.to_dict()`/역직렬화가 다른 모델과 동일하게 동작

- [x] `threedgrut/datasets/protocols.py`
  - [x] `Batch`에 `intrinsics_EquirectangularCameraModelParameters: Optional[dict] = None` 필드 추가
  - AC: 기존 데이터 경로 영향 없이 신규 필드 포함 시 모델 호출 가능

- [x] `threedgrut/datasets/utils.py`
  - [x] `equirectangular_camera_rays(width, height, ray_jitter=None)` 구현
    - 각 픽셀 중심 `(u=i+0.5, v=j+0.5)` → `phi = 2π*(u/W) - π`, `theta = π*(0.5 - v/H)`
    - `d = [cos(theta)*sin(phi), sin(theta), cos(theta)*cos(phi)]`, 초기버전은 yaw/pitch/roll 생략(후속 확장)
    - `rays_o = 0`, `rays_d = normalize(d)` 반환
  - AC: 작은 해상도에서의 단위 벡터 맵이 직관과 일치(+Z 중앙, +X 우측, +Y 아래)

- [x] `threedgut_tracer/tracer.py::__create_camera_parameters`
  - [x] ERP 분기 추가 및 `_3dgut_plugin.fromEquirectangularCameraModelParameters(...)` 호출
  - AC: ERP intrinsics 입력 시 바인딩이 호출되고 포즈 생성과 함께 반환

#### Python 사용 예
- ERP 파라미터(dict):
  - `resolution=np.array([W,H], dtype=np.int64)`, `shutter_type=ShutterType.GLOBAL`
- 데이터셋/배치 구성:
  - `sample["rays_ori"], sample["rays_dir"] = equirectangular_camera_rays(W, H)`
  - `sample["T_to_world"] = pose4x4`
  - `sample["intrinsics_EquirectangularCameraModelParameters"] = params.to_dict()`

### 구성(Hydra) 및 연결(선택 적용 경로: 렌더러 또는 데이터셋)
- [x] `configs/render/3dgut.yaml`
  - [x] 옵션 추가/반영됨:
    - `camera.model: [dataset|equirectangular]`
    - `camera.yaw_pitch_roll: [0.0, 0.0, 0.0]`
  - AC: 구성 로딩 시 충돌 없음

- [x] 적용 위치 결정 (택1)
  - [ ] 렌더러 경로: `threedgrut/render.py:Renderer.render_all()` 내, 배치 취득 후 ERP로 `rays_*`/`intrinsics_*` 대체
  - [x] 데이터셋 경로: `threedgrut/datasets/dataset_colmap.py:get_gpu_batch_with_intrinsics()`에서 구성에 따라 ERP 생성/주입
  - Note: 본 구현은 데이터셋 경로로 일원화되어 있으며, 렌더러 후처리 경로는 적용하지 않음

### 검증/회귀
- [ ] 플러그인 컴파일 및 단일 프레임 렌더
  - [x] 단위 벡터 케이스 검증: 중앙/사분면/극점 → (u,v) 기대치 확인
  - [x] φ 경계(±π) 근방 seam 미스/타일 누락 없음 확인
  - [ ] 전방 클램프(UT `ParticleMinSensorZ`)로 인한 동작 확인(문서화)
  - AC: 크래시/NaN 없이 이미지 저장, 타이밍 수집(`Tracer.timings`) 작동

### 결정을 요구하는 사항(초기 합의 필요)
- [ ] ERP 적용 지점: 렌더러 후처리 vs 데이터셋 공급(기본 제안: 데이터셋 경로로 일원화)
- [ ] φ 경계 처리 방식: 정규화([-π,π)) vs ε-클램프(성능/심미성 관점 결정)
- [ ] 완전 360° 지원 필요성: 전방 제약 유지(현재 기본) vs 향후 확장 항목으로 분리

### 후속(옵션)
- [ ] ERP 파라미터 확장: 수평/수직 크롭, fov 제한, yaw/pitch/roll 시간축 애니메이션
- [ ] 롤링셔터 시뮬레이션 고도화: ERP에서도 라인기반 shutter-time 샘플링 가시화 유틸

---

간단한 완성 정의(DoD)
- ERP 카메라 선택 시(구성 또는 데이터로), 3DGUT 경로에서 유효한 렌더 결과가 저장되고, 기존 Pinhole/Fisheye 시나리오에 회귀 영향이 없다.
- 최소 1개 씬에서 저해상도(예: 640×320) 렌더 성공, φ 경계/극점에서 아티팩트가 없음을 확인한다.
- 코드 레벨에서 ERP 관련 변경은 3DGUT 경로에 한정되며, 3DGRT와의 인터페이스가 오염되지 않는다.
