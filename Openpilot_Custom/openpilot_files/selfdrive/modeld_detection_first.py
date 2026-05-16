#!/usr/bin/env python3
import os

import cv2
from openpilot.system.hardware import TICI
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
if TICI:
  from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address
  os.environ['QCOM'] = '1'
else:
  os.environ['LLVM'] = '1'
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLContext
# from DepV2.DetAndDepthEstimate import inferenceDepth

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

VISION_PKL_PATH = Path(__file__).parent / 'models/driving_vision_tinygrad.pkl'
POLICY_PKL_PATH = Path(__file__).parent / 'models/driving_policy_tinygrad.pkl'
VISION_METADATA_PATH = Path(__file__).parent / 'models/driving_vision_metadata.pkl'
POLICY_METADATA_PATH = Path(__file__).parent / 'models/driving_policy_metadata.pkl'


def postProcessObjAndDepth(plotted_img, box_and_cls, org_depth):
    target = []
    for _ in box_and_cls:
        temp = {}
        cx, cy, w, h = [float(v) for v in _[:-1]]
        x_min = int(cx - w / 2)
        x_max = int(cx + w / 2)
        y_min = int(cy - h / 2)
        y_max = int(cy + h / 2)
        # print(x_min, x_max, y_min, y_max)
        print("org_depth->size: ", org_depth.shape)
        print(plotted_img.shape)
        ll = org_depth[int(y_min):int(y_max), int(x_min):int(x_max)]
        # cv2.rectangle(plotted_img, (x_min, y_min), (x_max, y_max), 255, 2)
        # print("ll: ", ll)
        # values = [x for x in ll if x is not None]
        # print("values in side: ", values)
        average_depth = (1.0 / np.mean(ll)) * 35.0
        temp["depth"] = average_depth
        temp["box"] = [float(v) for v in _[:-1]]
        temp["cls"] = [_[-1]]
        print(temp)
        target.append(temp)
        # print("cls:{} average depth:{} ".format(_[-1], average_depth))
        # text_x = x_min
        # text_y = y_min + 20
        # text = f"{average_depth:.2f}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # text_thickness = 2
        # text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        # color = (0, 255, 255)  # Green color (BGR format)
    return target



def inferenceByYOLOandPostProcessing(img):
    from ultralytics import YOLO
    model = YOLO("./yolo11x.pt")
    results = model(img)
    box_and_cls = postProcessingYOLO(results)
    return box_and_cls

def postProcessingYOLO(results, mode='xywh'):
    box_and_cls = []
    for result in results:
      xywh = result.boxes.xywh  # center-x, center-y, width, height
      # xywhn = result.boxes.xywhn  # normalized
      # xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
      # xyxyn = result.boxes.xyxyn  # normalized
      names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
      confs = result.boxes.conf  # confidence score of each box
      # print(names)
      # print(xywh)
      box = None
      if mode == "xywh":
        box = result.boxes.xywh
      elif mode == "xywhn":
        box = result.boxes.xywhn
      elif mode == "xyxy":
        box = result.boxes.xyxy
      elif mode == "xyxyn":
        box = result.boxes.xyxyn
      for _, cls in zip(box.tolist(), names):
        # print(_, cls)
        if cls in ["car", "truck", "bus"]:
          box_and_cls.append([*_, "car"])
        if cls in ["person", "people", "pedestrian"]:
          box_and_cls.append([*_, "pedestrian"])
    return box_and_cls

def recover_img(buf_main, saved_path):
  h = buf_main.height
  w = buf_main.width
  s = buf_main.stride  # bytes per row in both Y & UV planes
  uv_off = buf_main.uv_offset  # byte offset where UV plane starts in buf.data

  # 2. View the entire buffer as one flat uint8 array
  raw = np.frombuffer(buf_main.data, dtype=np.uint8)

  # 3. Extract & reshape the Y plane (h rows × s bytes), then crop to actual width
  y_plane = raw[0: h * s] \
    .reshape((h, s)) \
    [:, :w]

  # 4. Extract & reshape the UV plane ((h/2) rows × s bytes), then crop
  uv_plane = raw[uv_off: uv_off + (h // 2) * s] \
    .reshape((h // 2, s)) \
    [:, :w]

  # 5. Stack into NV12 layout and convert to BGR
  nv12 = np.vstack((y_plane, uv_plane))
  bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
  # cv2.imwrite(saved_path, bgr)
  return bgr

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class ModelState:
  frames: dict[str, DrivingModelFrame]
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, context: CLContext):
    self.frames = {
      'input_imgs': DrivingModelFrame(context, ModelConstants.TEMPORAL_SKIP),
      'big_input_imgs': DrivingModelFrame(context, ModelConstants.TEMPORAL_SKIP)
    }
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    self.full_features_buffer = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float32)
    self.full_desire = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float32)
    self.full_prev_desired_curv = np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float32)
    self.temporal_idxs = slice(-1-(ModelConstants.TEMPORAL_SKIP*(ModelConstants.INPUT_HISTORY_BUFFER_LEN-1)), None, ModelConstants.TEMPORAL_SKIP)

    # policy inputs
    self.numpy_inputs = {
      'desire': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float32),
      'traffic_convention': np.zeros((1, ModelConstants.TRAFFIC_CONVENTION_LEN), dtype=np.float32),
      'lateral_control_params': np.zeros((1, ModelConstants.LATERAL_CONTROL_PARAMS_LEN), dtype=np.float32),
      'prev_desired_curv': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float32),
      'features_buffer': np.zeros((1, ModelConstants.INPUT_HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float32),
    }

    with open(VISION_METADATA_PATH, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_input_shapes =  vision_metadata['input_shapes']
      print(self.vision_input_shapes)
      self.vision_output_slices = vision_metadata['output_slices']
      vision_output_size = vision_metadata['output_shapes']['outputs'][1]

    with open(POLICY_METADATA_PATH, 'rb') as f:
      policy_metadata = pickle.load(f)
      self.policy_input_shapes =  policy_metadata['input_shapes']
      self.policy_output_slices = policy_metadata['output_slices']
      policy_output_size = policy_metadata['output_shapes']['outputs'][1]

    # img buffers are managed in openCL transform code
    self.vision_inputs: dict[str, Tensor] = {}
    self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
    self.policy_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}
    self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
    self.parser = Parser()

    with open(VISION_PKL_PATH, "rb") as f:
      self.vision_run = pickle.load(f)

    with open(POLICY_PKL_PATH, "rb") as f:
      self.policy_run = pickle.load(f)

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs



  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: dict[str, np.ndarray], prepare_only: bool, file_name1: str,file_name2: str, bgr:np.ndarray) -> dict[str, np.ndarray] | None:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire'][0] = 0
    new_desire = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']
    # print(self.frames.keys())
    # print(self.frames['input_imgs'].prepare())

    self.full_desire[0,:-1] = self.full_desire[0,1:]
    self.full_desire[0,-1] = new_desire
    self.numpy_inputs['desire'][:] = self.full_desire.reshape((1,ModelConstants.INPUT_HISTORY_BUFFER_LEN,ModelConstants.TEMPORAL_SKIP,-1)).max(axis=2)

    self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.numpy_inputs['lateral_control_params'][:] = inputs['lateral_control_params']
    imgs_cl = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
               'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}

    if TICI:
      # The imgs tensors are backed by opencl memory, only need init once
      for key in imgs_cl:
        if key not in self.vision_inputs:
          self.vision_inputs[key] = qcom_tensor_from_opencl_address(imgs_cl[key].mem_address, self.vision_input_shapes[key], dtype=dtypes.uint8)
    else:
      for key in imgs_cl:
        frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.vision_input_shapes[key])
        self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()

    if prepare_only:
      return None
    print(self.vision_inputs.keys())
    print(self.vision_inputs['input_imgs'].shape, self.vision_inputs['big_input_imgs'].shape)
    self.vision_output = self.vision_run(**self.vision_inputs).numpy().flatten()
    print(self.vision_output.shape)
    print(type(self.vision_output))
    # cv2.imwrite(file_name, self.vision_output)
    for key in self.policy_inputs.keys():
      print(self.policy_inputs[key].shape)
    vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(self.vision_output, self.vision_output_slices))

    self.full_features_buffer[0,:-1] = self.full_features_buffer[0,1:]
    print(vision_outputs_dict['hidden_state'][0, :].shape)
    print("saving img feature !!!!!")
    cv2.imwrite(file_name1, vision_outputs_dict['hidden_state'][0, :])
    cv2.imwrite(file_name2, bgr)
    self.full_features_buffer[0,-1] = vision_outputs_dict['hidden_state'][0, :]
    self.numpy_inputs['features_buffer'][:] = self.full_features_buffer[0, self.temporal_idxs]
    print(self.policy_inputs.keys())

    self.policy_output = self.policy_run(**self.policy_inputs).numpy().flatten()
    print("runs here !!!!!!!!!!!")
    print(self.policy_output.shape)
    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(self.policy_output, self.policy_output_slices))

    # TODO model only uses last value now
    self.full_prev_desired_curv[0,:-1] = self.full_prev_desired_curv[0,1:]
    self.full_prev_desired_curv[0,-1,:] = policy_outputs_dict['desired_curvature'][0, :]
    self.numpy_inputs['prev_desired_curv'][:] = self.full_prev_desired_curv[0, self.temporal_idxs]

    combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
    if SEND_RAW_PRED:
      combined_outputs_dict['raw_pred'] = np.concatenate([self.vision_output.copy(), self.policy_output.copy()])

    return combined_outputs_dict


def main(demo=False):
  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)

  cloudlog.warning("setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("CL context ready; loading model")
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")
  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  steer_delay = CP.steerActuatorDelay + .2
  # print("111111111111111111")
  DH = DesireHelper()
  count = 0 #
  # Create directories for saving test dataset
  dir = "./test_dataset/new/" #MODIFY for saving
  os.makedirs(dir + "features/", exist_ok=True)
  os.makedirs(dir + "raw/", exist_ok=True)
  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lateral_control_params = np.array([v_ego, steer_delay], dtype=np.float32)
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    inputs:dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'lateral_control_params': lateral_control_params,
      }

    mt1 = time.perf_counter()
    mid = f"{count:06d}"
    img_feature_name = dir + "features/"+ mid + ".png"
    bgr = recover_img(buf_main, dir + "raw/" + mid + ".png")
    img_name = dir + "raw/" + mid + ".png"
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only, file_name1=img_feature_name, file_name2=img_name, bgr=bgr)
    # mt2 = time.perf_counter()
    count += 1
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      fill_model_msg(drivingdata_send, modelv2_send, model_output, v_ego, steer_delay,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id



if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise
